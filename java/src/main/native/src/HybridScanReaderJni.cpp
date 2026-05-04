/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "jni_compiled_expr.hpp"
#include "jni_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace exp_pq = cudf::io::parquet::experimental;
using cudf::io::text::byte_range_info;

/**
 * @brief Wrapper that owns the C++ hybrid_scan_reader plus the parquet_reader_options that
 *        most reader methods need to be supplied alongside the call. Keeping them together
 *        avoids the need for the caller (Java) to round-trip the options across JNI on each
 *        call.
 */
struct hybrid_scan_reader_wrapper {
  cudf::io::parquet_reader_options options;
  std::unique_ptr<exp_pq::hybrid_scan_reader> reader;

  hybrid_scan_reader_wrapper(cudf::host_span<uint8_t const> footer_bytes,
                             cudf::io::parquet_reader_options opts)
    : options(std::move(opts)),
      reader(std::make_unique<exp_pq::hybrid_scan_reader>(footer_bytes, options))
  {
  }
};

/**
 * @brief Holds the result of a planned host-to-device copy: the actual device memory and a
 *        list of device spans into that memory matching the requested byte ranges.
 */
struct planned_copy_result {
  std::vector<rmm::device_buffer> device_buffers;
  std::vector<cudf::device_span<uint8_t const>> spans;
};

bool are_ranges_contiguous(std::vector<byte_range_info> const& ranges)
{
  if (ranges.size() <= 1) { return true; }
  for (size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i - 1].offset() + ranges[i - 1].size() != ranges[i].offset()) { return false; }
  }
  return true;
}

planned_copy_result plan_and_copy_ranges(uint8_t const* host_ptr,
                                         std::vector<byte_range_info> const& ranges,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  planned_copy_result result;
  if (ranges.empty()) { return result; }

  if (are_ranges_contiguous(ranges)) {
    auto const first_offset = ranges.front().offset();
    auto const last_range   = ranges.back();
    auto const total_size   = (last_range.offset() + last_range.size()) - first_offset;
    rmm::device_buffer coalesced_buf(total_size, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(coalesced_buf.data(),
                                  host_ptr + first_offset,
                                  total_size,
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    result.spans.reserve(ranges.size());
    auto const* base_ptr = static_cast<uint8_t const*>(coalesced_buf.data());
    for (auto const& range : ranges) {
      auto const offset_in_buffer = range.offset() - first_offset;
      result.spans.emplace_back(base_ptr + offset_in_buffer, range.size());
    }
    result.device_buffers.emplace_back(std::move(coalesced_buf));
  } else {
    result.device_buffers.reserve(ranges.size());
    for (auto const& range : ranges) {
      rmm::device_buffer dev_buf(range.size(), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(dev_buf.data(),
                                    host_ptr + range.offset(),
                                    range.size(),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      result.device_buffers.emplace_back(std::move(dev_buf));
    }
    result.spans.reserve(result.device_buffers.size());
    for (auto const& buf : result.device_buffers) {
      result.spans.emplace_back(static_cast<uint8_t const*>(buf.data()), buf.size());
    }
  }
  return result;
}

/**
 * @brief Build a parquet_reader_options from the supplied JNI args. The footer is provided
 *        separately because the hybrid_scan_reader does not consume it via the options.
 */
cudf::io::parquet_reader_options build_options(JNIEnv* env,
                                               jlong filter_handle,
                                               jobjectArray j_column_names,
                                               jbooleanArray j_read_binary_as_string,
                                               jint time_unit_type_id)
{
  // The hybrid_scan_reader's options builder is constructed without a source_info because
  // the reader works on already-parsed footer bytes (and on byte ranges fetched separately).
  cudf::io::parquet_reader_options_builder builder;

  cudf::jni::native_jstringArray names(env, j_column_names);
  if (!names.is_null() && names.size() > 0) {
    builder = builder.column_names(names.as_cpp_vector());
  }

  // Translate Java's per-column "read binary as string" flags into the C++ schema override
  // hooks. The reader_column_schema mechanism lets callers force binary→string conversion
  // for the i-th projected column.
  cudf::jni::native_jbooleanArray binary_as_str(env, j_read_binary_as_string);
  if (!binary_as_str.is_null() && binary_as_str.size() > 0) {
    std::vector<cudf::io::reader_column_schema> schemas;
    schemas.reserve(binary_as_str.size());
    for (int i = 0; i < binary_as_str.size(); ++i) {
      cudf::io::reader_column_schema s;
      s.set_convert_binary_to_strings(static_cast<bool>(binary_as_str[i]));
      schemas.emplace_back(std::move(s));
    }
    builder = builder.set_column_schema(std::move(schemas));
    binary_as_str.cancel();
  }

  auto opts =
    builder
      .convert_strings_to_categories(false)
      .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(time_unit_type_id)))
      .ignore_missing_columns(true)
      .build();

  if (filter_handle != 0) {
    auto const* filter_expr =
      reinterpret_cast<cudf::jni::ast::compiled_expr const*>(filter_handle);
    opts.set_filter(filter_expr->get_top_expression());
  }

  return opts;
}

/**
 * @brief Convert a Java int[] of row group indices into a host_span<size_type const>. The
 *        wrapper owns a vector that backs the span; capture it as a value to avoid dangling.
 */
struct row_group_span_holder {
  std::vector<cudf::size_type> storage;
  cudf::host_span<cudf::size_type const> span() const { return {storage.data(), storage.size()}; }
};

row_group_span_holder make_row_group_span(JNIEnv* env, jintArray j_row_groups)
{
  row_group_span_holder h;
  cudf::jni::native_jintArray arr(env, j_row_groups);
  h.storage.reserve(arr.size());
  for (int i = 0; i < arr.size(); ++i) {
    h.storage.push_back(static_cast<cudf::size_type>(arr[i]));
  }
  arr.cancel();
  return h;
}

/**
 * @brief Convert a vector<byte_range_info> into a packed jlongArray laid out as
 *        [off0, len0, off1, len1, ...].
 */
jlongArray ranges_to_jlong_array(JNIEnv* env, std::vector<byte_range_info> const& ranges)
{
  auto result = env->NewLongArray(ranges.size() * 2);
  if (result == nullptr) { return nullptr; }
  if (ranges.empty()) { return result; }
  std::vector<jlong> data;
  data.reserve(ranges.size() * 2);
  for (auto const& r : ranges) {
    data.push_back(static_cast<jlong>(r.offset()));
    data.push_back(static_cast<jlong>(r.size()));
  }
  env->SetLongArrayRegion(result, 0, data.size(), data.data());
  return result;
}

jintArray sizes_to_jint_array(JNIEnv* env, std::vector<cudf::size_type> const& vals)
{
  auto result = env->NewIntArray(vals.size());
  if (result == nullptr) { return nullptr; }
  if (vals.empty()) { return result; }
  // jint is int32_t; size_type is also int32_t. Static-cast just to be explicit.
  std::vector<jint> j(vals.begin(), vals.end());
  env->SetIntArrayRegion(result, 0, j.size(), j.data());
  return result;
}

std::vector<cudf::device_span<uint8_t const>> make_device_spans(JNIEnv* env,
                                                                jlongArray j_addrs,
                                                                jlongArray j_lens)
{
  cudf::jni::native_jlongArray addrs(env, j_addrs);
  cudf::jni::native_jlongArray lens(env, j_lens);
  std::vector<cudf::device_span<uint8_t const>> out;
  out.reserve(addrs.size());
  for (int i = 0; i < addrs.size(); ++i) {
    out.emplace_back(reinterpret_cast<uint8_t const*>(addrs[i]), static_cast<size_t>(lens[i]));
  }
  addrs.cancel();
  lens.cancel();
  return out;
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_createFromFooter(JNIEnv* env,
                                                                   jclass,
                                                                   jlong footer_address,
                                                                   jlong footer_length,
                                                                   jlong filter_handle,
                                                                   jobjectArray j_column_names,
                                                                   jbooleanArray j_binary_as_str,
                                                                   jint time_unit_type_id)
{
  JNI_NULL_CHECK(env, footer_address, "footer address is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto opts = build_options(
      env, filter_handle, j_column_names, j_binary_as_str, time_unit_type_id);
    auto const* footer_ptr = reinterpret_cast<uint8_t const*>(footer_address);
    cudf::host_span<uint8_t const> footer_bytes{footer_ptr, static_cast<size_t>(footer_length)};
    auto wrapper = std::make_unique<hybrid_scan_reader_wrapper>(footer_bytes, std::move(opts));
    return reinterpret_cast<jlong>(wrapper.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_destroy(JNIEnv* env, jclass, jlong handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
  }
  JNI_CATCH(env, );
}

// ----------------------------------------------------------------------
// Metadata
// ----------------------------------------------------------------------

JNIEXPORT jobject JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_parquetMetadata(JNIEnv* env,
                                                                  jclass,
                                                                  jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto md       = wrapper->reader->parquet_metadata();

    jclass cls = env->FindClass("ai/rapids/cudf/experimental/FileMetaData");
    if (cls == nullptr) { return nullptr; }
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(IJLjava/lang/String;)V");
    if (ctor == nullptr) { return nullptr; }

    jstring created_by = env->NewStringUTF(md.created_by.c_str());
    if (created_by == nullptr) { return nullptr; }

    return env->NewObject(cls,
                          ctor,
                          static_cast<jint>(md.version),
                          static_cast<jlong>(md.num_rows),
                          created_by);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_pageIndexByteRange(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto range    = wrapper->reader->page_index_byte_range();
    jlong vals[2] = {static_cast<jlong>(range.offset()), static_cast<jlong>(range.size())};
    auto result   = env->NewLongArray(2);
    if (result == nullptr) { return nullptr; }
    env->SetLongArrayRegion(result, 0, 2, vals);
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_setupPageIndex(JNIEnv* env,
                                                                 jclass,
                                                                 jlong handle,
                                                                 jlong buffer_address,
                                                                 jlong buffer_length)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );
  JNI_NULL_CHECK(env, buffer_address, "page index buffer is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper        = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto const* host_ptr = reinterpret_cast<uint8_t const*>(buffer_address);
    cudf::host_span<uint8_t const> bytes{host_ptr, static_cast<size_t>(buffer_length)};
    wrapper->reader->setup_page_index(bytes);
  }
  JNI_CATCH(env, );
}

// ----------------------------------------------------------------------
// Row group enumeration
// ----------------------------------------------------------------------

JNIEXPORT jintArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_allRowGroups(JNIEnv* env, jclass, jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto rgs      = wrapper->reader->all_row_groups(wrapper->options);
    return sizes_to_jint_array(env, rgs);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_totalRowsInRowGroups(JNIEnv* env,
                                                                       jclass,
                                                                       jlong handle,
                                                                       jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", 0);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto total    = wrapper->reader->total_rows_in_row_groups(holder.span());
    return static_cast<jlong>(total);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_resetColumnSelection(JNIEnv* env,
                                                                       jclass,
                                                                       jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    wrapper->reader->reset_column_selection();
  }
  JNI_CATCH(env, );
}

// ----------------------------------------------------------------------
// Filtering
// ----------------------------------------------------------------------

JNIEXPORT jintArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_filterRowGroupsWithStats(JNIEnv* env,
                                                                           jclass,
                                                                           jlong handle,
                                                                           jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto filtered = wrapper->reader->filter_row_groups_with_stats(
      holder.span(), wrapper->options, cudf::get_default_stream());
    return sizes_to_jint_array(env, filtered);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_secondaryFiltersByteRanges(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto [bloom, dict] =
      wrapper->reader->secondary_filters_byte_ranges(holder.span(), wrapper->options);
    // Pack as [numBloom, bloom_o0, bloom_s0, ..., dict_o0, dict_s0, ...]
    auto const total_len = 1 + (bloom.size() + dict.size()) * 2;
    auto result          = env->NewLongArray(total_len);
    if (result == nullptr) { return nullptr; }
    std::vector<jlong> data;
    data.reserve(total_len);
    data.push_back(static_cast<jlong>(bloom.size()));
    for (auto const& r : bloom) {
      data.push_back(static_cast<jlong>(r.offset()));
      data.push_back(static_cast<jlong>(r.size()));
    }
    for (auto const& r : dict) {
      data.push_back(static_cast<jlong>(r.offset()));
      data.push_back(static_cast<jlong>(r.size()));
    }
    env->SetLongArrayRegion(result, 0, data.size(), data.data());
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jintArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_filterRowGroupsWithBloomFilters(
  JNIEnv* env,
  jclass,
  jlong handle,
  jlongArray j_addrs,
  jlongArray j_lens,
  jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto spans    = make_device_spans(env, j_addrs, j_lens);
    auto filtered = wrapper->reader->filter_row_groups_with_bloom_filters(
      spans, holder.span(), wrapper->options, cudf::get_default_stream());
    return sizes_to_jint_array(env, filtered);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jintArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_filterRowGroupsWithDictionaryPages(
  JNIEnv* env,
  jclass,
  jlong handle,
  jlongArray j_addrs,
  jlongArray j_lens,
  jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto spans    = make_device_spans(env, j_addrs, j_lens);
    auto filtered = wrapper->reader->filter_row_groups_with_dictionary_pages(
      spans, holder.span(), wrapper->options, cudf::get_default_stream());
    return sizes_to_jint_array(env, filtered);
  }
  JNI_CATCH(env, nullptr);
}

// ----------------------------------------------------------------------
// Byte ranges
// ----------------------------------------------------------------------

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_filterColumnChunksByteRanges(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto ranges =
      wrapper->reader->filter_column_chunks_byte_ranges(holder.span(), wrapper->options);
    return ranges_to_jlong_array(env, ranges);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_payloadColumnChunksByteRanges(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto ranges =
      wrapper->reader->payload_column_chunks_byte_ranges(holder.span(), wrapper->options);
    return ranges_to_jlong_array(env, ranges);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_allColumnChunksByteRanges(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle,
                                                                            jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto ranges =
      wrapper->reader->all_column_chunks_byte_ranges(holder.span(), wrapper->options);
    return ranges_to_jlong_array(env, ranges);
  }
  JNI_CATCH(env, nullptr);
}

// ----------------------------------------------------------------------
// Single-shot materialize
// ----------------------------------------------------------------------

// Returns: [row_mask_col_handle, filter_table_col0_handle, ..., filter_table_colN_handle]
JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_materializeFilterColumnsWithKind(
  JNIEnv* env,
  jclass,
  jlong handle,
  jintArray j_row_groups,
  jlongArray j_addrs,
  jlongArray j_lens,
  jboolean use_data_page_mask,
  jboolean all_true)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto spans    = make_device_spans(env, j_addrs, j_lens);
    auto stream   = cudf::get_default_stream();
    auto mr       = cudf::get_current_device_resource_ref();
    // Build the owned row mask according to the requested kind.
    std::unique_ptr<cudf::column> row_mask_col;
    if (all_true) {
      row_mask_col = wrapper->reader->build_all_true_row_mask(holder.span(), stream, mr);
    } else {
      row_mask_col = wrapper->reader->build_row_mask_with_page_index_stats(
        holder.span(), wrapper->options, stream, mr);
    }
    auto mut_view = row_mask_col->mutable_view();
    auto mode     = use_data_page_mask ? exp_pq::use_data_page_mask::YES
                                       : exp_pq::use_data_page_mask::NO;
    auto result   = wrapper->reader->materialize_filter_columns(
      holder.span(), spans, mut_view, mode, wrapper->options, stream, mr);
    // Pack: [row_mask_handle, table_col0, ..., table_colN]
    auto table_handles = cudf::jni::convert_table_for_return(env, result.tbl);
    cudf::jni::native_jlongArray table_arr(env, table_handles);
    jsize n_table_cols = table_arr.size();
    jlongArray out = env->NewLongArray(1 + n_table_cols);
    if (out == nullptr) { return nullptr; }
    jlong row_mask_handle = cudf::jni::release_as_jlong(std::move(row_mask_col));
    env->SetLongArrayRegion(out, 0, 1, &row_mask_handle);
    env->SetLongArrayRegion(out, 1, n_table_cols, table_arr.data());
    return out;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_materializePayloadColumns(
  JNIEnv* env,
  jclass,
  jlong handle,
  jintArray j_row_groups,
  jlongArray j_addrs,
  jlongArray j_lens,
  jlong row_mask_view_handle,
  jboolean use_data_page_mask)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_NULL_CHECK(env, row_mask_view_handle, "row mask view handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper  = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder    = make_row_group_span(env, j_row_groups);
    auto spans     = make_device_spans(env, j_addrs, j_lens);
    auto* row_mask = reinterpret_cast<cudf::column_view const*>(row_mask_view_handle);
    auto mode      = use_data_page_mask ? exp_pq::use_data_page_mask::YES
                                        : exp_pq::use_data_page_mask::NO;
    auto result    = wrapper->reader->materialize_payload_columns(
      holder.span(),
      spans,
      *row_mask,
      mode,
      wrapper->options,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_materializeAllColumns(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle,
                                                                        jintArray j_row_groups,
                                                                        jlongArray j_addrs,
                                                                        jlongArray j_lens)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto spans    = make_device_spans(env, j_addrs, j_lens);
    auto result   = wrapper->reader->materialize_all_columns(
      holder.span(),
      spans,
      wrapper->options,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

// ----------------------------------------------------------------------
// Chunked materialize
// ----------------------------------------------------------------------

// Returns owned row mask column handle; caller threads it into materializeFilterColumnsChunk.
JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_setupChunkingForFilterColumnsWithKind(
  JNIEnv* env,
  jclass,
  jlong handle,
  jlong chunk_read_limit,
  jlong pass_read_limit,
  jintArray j_row_groups,
  jboolean use_data_page_mask,
  jboolean all_true,
  jlongArray j_addrs,
  jlongArray j_lens)
{
  JNI_NULL_CHECK(env, handle, "handle is null", 0);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto spans    = make_device_spans(env, j_addrs, j_lens);
    auto stream   = cudf::get_default_stream();
    auto mr       = cudf::get_current_device_resource_ref();
    // Build the owned row mask according to the requested kind.
    std::unique_ptr<cudf::column> row_mask_col;
    if (all_true) {
      row_mask_col = wrapper->reader->build_all_true_row_mask(holder.span(), stream, mr);
    } else {
      row_mask_col = wrapper->reader->build_row_mask_with_page_index_stats(
        holder.span(), wrapper->options, stream, mr);
    }
    auto mode = use_data_page_mask ? exp_pq::use_data_page_mask::YES
                                   : exp_pq::use_data_page_mask::NO;
    // Pass a read-only view during setup; the owned column is returned to Java.
    wrapper->reader->setup_chunking_for_filter_columns(
      static_cast<std::size_t>(chunk_read_limit),
      static_cast<std::size_t>(pass_read_limit),
      holder.span(),
      row_mask_col->view(),
      mode,
      spans,
      wrapper->options,
      stream,
      mr);
    return cudf::jni::release_as_jlong(std::move(row_mask_col));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_materializeFilterColumnsChunk(
  JNIEnv* env, jclass, jlong handle, jlong row_mask_column_handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, row_mask_column_handle, "row mask handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper  = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto* row_mask = reinterpret_cast<cudf::column*>(row_mask_column_handle);
    auto mut_view  = row_mask->mutable_view();
    auto result    = wrapper->reader->materialize_filter_columns_chunk(mut_view);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_setupChunkingForPayloadColumns(
  JNIEnv* env,
  jclass,
  jlong handle,
  jlong chunk_read_limit,
  jlong pass_read_limit,
  jintArray j_row_groups,
  jlong row_mask_view_handle,
  jboolean use_data_page_mask,
  jlongArray j_addrs,
  jlongArray j_lens)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", );
  JNI_NULL_CHECK(env, row_mask_view_handle, "row mask view handle is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper  = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder    = make_row_group_span(env, j_row_groups);
    auto spans     = make_device_spans(env, j_addrs, j_lens);
    auto* row_mask = reinterpret_cast<cudf::column_view const*>(row_mask_view_handle);
    auto mode      = use_data_page_mask ? exp_pq::use_data_page_mask::YES
                                        : exp_pq::use_data_page_mask::NO;
    wrapper->reader->setup_chunking_for_payload_columns(
      static_cast<std::size_t>(chunk_read_limit),
      static_cast<std::size_t>(pass_read_limit),
      holder.span(),
      *row_mask,
      mode,
      spans,
      wrapper->options,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_materializePayloadColumnsChunk(
  JNIEnv* env, jclass, jlong handle, jlong row_mask_view_handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, row_mask_view_handle, "row mask view handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper  = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto* row_mask = reinterpret_cast<cudf::column_view const*>(row_mask_view_handle);
    auto result    = wrapper->reader->materialize_payload_columns_chunk(*row_mask);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_setupChunkingForAllColumns(
  JNIEnv* env,
  jclass,
  jlong handle,
  jlong chunk_read_limit,
  jlong pass_read_limit,
  jintArray j_row_groups,
  jlongArray j_addrs,
  jlongArray j_lens)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto spans    = make_device_spans(env, j_addrs, j_lens);
    wrapper->reader->setup_chunking_for_all_columns(
      static_cast<std::size_t>(chunk_read_limit),
      static_cast<std::size_t>(pass_read_limit),
      holder.span(),
      spans,
      wrapper->options,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_materializeAllColumnsChunk(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto result   = wrapper->reader->materialize_all_columns_chunk();
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jboolean JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_hasNextTableChunk(JNIEnv* env,
                                                                    jclass,
                                                                    jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", JNI_FALSE);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    return wrapper->reader->has_next_table_chunk() ? JNI_TRUE : JNI_FALSE;
  }
  JNI_CATCH(env, JNI_FALSE);
}

JNIEXPORT jobjectArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_constructRowGroupPasses(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups, jlong pass_read_limit)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto passes   = wrapper->reader->construct_row_group_passes(
      holder.span(), static_cast<std::size_t>(pass_read_limit));
    jclass int_array_cls = env->FindClass("[I");
    if (int_array_cls == nullptr) { return nullptr; }
    auto outer = env->NewObjectArray(passes.size(), int_array_cls, nullptr);
    if (outer == nullptr) { return nullptr; }
    for (size_t i = 0; i < passes.size(); ++i) {
      auto inner = sizes_to_jint_array(env, passes[i]);
      if (inner == nullptr) { return nullptr; }
      env->SetObjectArrayElement(outer, static_cast<jsize>(i), inner);
      env->DeleteLocalRef(inner);
    }
    return outer;
  }
  JNI_CATCH(env, nullptr);
}

// ----------------------------------------------------------------------
// Convenience: full file from a single host buffer
// ----------------------------------------------------------------------

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_experimental_HybridScanReader_materializeFromBuffer(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle,
                                                                        jlong buffer_address,
                                                                        jlong buffer_length)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, buffer_address, "buffer address is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper        = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto const* host_ptr = reinterpret_cast<uint8_t const*>(buffer_address);
    auto const stream    = cudf::get_default_stream();
    auto const mr        = cudf::get_current_device_resource_ref();

    auto all_rgs = wrapper->reader->all_row_groups(wrapper->options);
    cudf::host_span<cudf::size_type const> rg_span{all_rgs};

    bool const has_filter = wrapper->options.get_filter().has_value();

    // Optionally fetch + materialize the filter columns first to update the row mask.
    planned_copy_result filter_copy;
    if (has_filter) {
      auto filter_ranges =
        wrapper->reader->filter_column_chunks_byte_ranges(rg_span, wrapper->options);
      filter_copy = plan_and_copy_ranges(host_ptr, filter_ranges, stream, mr);
    }

    // Build an all-true row mask.
    auto const total_rows = wrapper->reader->total_rows_in_row_groups(rg_span);
    auto true_scalar      = cudf::make_fixed_width_scalar<bool>(true, stream);
    auto row_mask         = cudf::make_column_from_scalar(*true_scalar, total_rows, stream, mr);
    auto row_mask_view    = row_mask->mutable_view();

    cudf::io::table_with_metadata filter_result;
    if (has_filter) {
      filter_result = wrapper->reader->materialize_filter_columns(
        rg_span,
        filter_copy.spans,
        row_mask_view,
        exp_pq::use_data_page_mask::NO,
        wrapper->options,
        stream,
        mr);
    }

    // Always read payload (the C++ reader internally handles "no payload columns" gracefully).
    auto payload_ranges =
      wrapper->reader->payload_column_chunks_byte_ranges(rg_span, wrapper->options);
    auto payload_copy = plan_and_copy_ranges(host_ptr, payload_ranges, stream, mr);

    auto const mode = has_filter ? exp_pq::use_data_page_mask::YES
                                 : exp_pq::use_data_page_mask::NO;
    auto payload_result = wrapper->reader->materialize_payload_columns(
      rg_span,
      payload_copy.spans,
      row_mask->view(),
      mode,
      wrapper->options,
      stream,
      mr);

    // If there were no filter columns, we can return the payload table directly.
    if (!filter_result.tbl) {
      return cudf::jni::convert_table_for_return(env, payload_result.tbl);
    }

    // Otherwise, merge filter and payload columns in the order returned by the reader.
    // The hybrid_scan_reader internally tracks "filter" vs "payload" but the user-visible
    // schema is the union; for the convenience method we just append payload after filter.
    auto filter_cols  = filter_result.tbl->release();
    auto payload_cols = payload_result.tbl->release();
    std::vector<std::unique_ptr<cudf::column>> merged;
    merged.reserve(filter_cols.size() + payload_cols.size());
    for (auto& c : filter_cols) {
      merged.emplace_back(std::move(c));
    }
    for (auto& c : payload_cols) {
      merged.emplace_back(std::move(c));
    }
    auto merged_table = std::make_unique<cudf::table>(std::move(merged));
    return cudf::jni::convert_table_for_return(env, std::move(merged_table));
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
