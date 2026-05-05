/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "hybrid_scan_jni_internal.hpp"
#include "jni_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <memory>
#include <utility>
#include <vector>

using namespace cudf::jni::hybrid_scan;

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
