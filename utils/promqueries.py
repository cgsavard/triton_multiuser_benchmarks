import prometheus_api_client
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.metric_range_df import MetricRangeDataFrame
from prometheus_api_client.metric_snapshot_df import MetricSnapshotDataFrame
from prometheus_api_client.metrics_list import MetricsList
from prometheus_api_client.utils import parse_datetime
import pandas as pd

prom = PrometheusConnect(url="http://lsdataitb.fnal.gov:9009/prometheus", disable_ssl=True)


# A function for getting queries of many GPU and Triton server metrics. Inputs are a list of timestamp tuples,
# which can be parsed by the prometheus_api_client.utils.parse_datetime function. This can understand timestamps formatted like
# "2023-03-30 at 16:00:00 MDT"
# The step is the 'time-window' over which each query will be divided. This should be ~4x as long as the longest frequency for metric-gather
def get_all_queries(timestamp_tuples, step):
    # A dictionary for our results
    results = {}
    # Tuples of the queries we'll make, for debugging and info
    queries = []
    
    # Some queries are best created after understanding which unique models+version have been run in the triton servers
    # and which GPU instances have been active. These are then used to formulate model/version-specific and GPU-specific stats
    unique_model_versions = None
    unique_gpu_instances = None
    
    #Basic queries. Some of them are used as proxies to figure out the unqique queries to make later, like the "gpu_tensor_util" below
    for key, query in {
        "num_instances": "count((sum by(pod) (delta(nv_inference_request_success["+step+"]))) > 0)",
        "inf_rate_net":"sum (rate(nv_inference_count["+step+"]))",
        "inf_rate_bypod":"sum by(pod) (rate(nv_inference_count["+step+"]))",
        "inf_rate":"sum by(model, version, pod) (rate(nv_inference_count["+step+"]))",
        "inf_cache_hit_rate":"sum by(model, version, pod) (rate(nv_cache_num_hits_per_model["+step+"]))",
        "inf_reqs_net":"sum(rate(nv_inference_request_success["+step+"]))",
        "inf_reqs_bypod":"sum by(pod) (rate(nv_inference_request_success["+step+"]))",
        "inf_reqs":"sum by(model, version, pod) (rate(nv_inference_request_success["+step+"]))",
        "inf_req_dur_net": "avg (delta(nv_inference_request_duration_us["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        "inf_que_dur_net": "avg (delta(nv_inference_queue_duration_us["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        "inf_inp_dur_net": "avg (delta(nv_inference_compute_input_duration_us["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        "inf_inf_dur_net": "avg (delta(nv_inference_compute_infer_duration_us["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        "inf_out_dur_net": "avg (delta(nv_inference_compute_output_duration_us["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        "inf_req_dur": "avg by(model, version, pod) (delta(nv_inference_request_duration_us["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        "inf_que_dur": "avg by(model, version, pod) (delta(nv_inference_queue_duration_us["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        "inf_inp_dur": "avg by(model, version, pod) (delta(nv_inference_compute_input_duration_us["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        "inf_inf_dur": "avg by(model, version, pod) (delta(nv_inference_compute_infer_duration_us["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        "inf_out_dur": "avg by(model, version, pod) (delta(nv_inference_compute_output_duration_us["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        "gpu_tensor_util": "sum by(device,GPU_I_ID,instance) (avg_over_time (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{exported_container='triton',exported_namespace='triton',prometheus_replica='prometheus-k8s-0'}["+step+"]))",
        "gpu_dram_util": "sum by(device,GPU_I_ID,instance) (avg_over_time (DCGM_FI_PROF_DRAM_ACTIVE{exported_container='triton',exported_namespace='triton',prometheus_replica='prometheus-k8s-0'}["+step+"]))",
        #"inf_cache_hits": "avg by(model, version, pod) (delta(nv_cache_num_hits_per_model["+step+"])/(1+1000000*delta(nv_inference_request_success["+step+"])))",
        }.items():
        # Build an empty list for these results; after iterating through all the timestamp pairs, they'll be concatenated together
        results[key] = []
        # Log the queries, as they're easier to parse after being resolved fully
        queries.append((key, query))
        # This function executes a query for each timestamp pair, for each key:query
        for st, et in timestamp_tuples:
            test_inp = prom.custom_query_range(
                query=query,
                start_time=st,
                end_time=et,
                step=step
            )
            # Queries are converted to a pandas dataframe
            df = MetricRangeDataFrame(test_inp)
            results[key].append(df)
        # Dataframes are concatenated together along the time (index value) axis
        results[key] = pd.concat(results[key], axis=0)
        
        # If we've performed a query that stores model/version info and GPU instance info, respectively, we can 
        # Create a set of unique ones for the next two sets of queries
        if unique_model_versions is None and hasattr(results[key], "model") and hasattr(results[key], "version"):
            unique_model_versions = set((results[key].model+"/"+results[key].version).values)
        # At the EAF, the device ('nvidiaX' where X is 0...4 for example), GPU instance ID (enumeration)
        # and the instance (IP address of host machine) are sufficient to make a unique identifier
        if unique_gpu_instances is None and hasattr(results[key], "GPU_I_ID"):
            unique_gpu_instances = set((results[key].device+"/"+results[key].GPU_I_ID+"/"+results[key].instance).values)
    # Here we build the model-specific queries, getting both the number of unique number of Triton instances that served 
    # inference requests for this model, ad well as the inference rate of that model across all Triton instances active per time step
    model_queries = {"num_instances_"+model_version: "count((sum by(pod) (delta(nv_inference_request_success{model='"+
                     model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'}["+step+"]))) > 0)"
                     for model_version in unique_model_versions}
    model_queries.update(
        {"inf_rate_"+model_version: "sum (rate(nv_inference_count{model='"+
         model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'}["+step+"]))"
         for model_version in unique_model_versions})
    for key, query in model_queries.items():
        queries.append((key, query))
        results[key] = []
        for st, et in timestamp_tuples:
            test_inp = prom.custom_query_range(
                query=query,
                start_time=st,
                end_time=et,
                step=step
            )
            # The query could be empty, as a model only served in a portion of the total timerange could be inactive in some timestamp-pairs.
            # We will deal with broadcasting these dataframes with missing values later
            if len(test_inp) > 0:
                df = MetricRangeDataFrame(test_inp)
                results[key].append(df)
        if len(results[key]) > 0:
            results[key] = pd.concat(results[key], axis=0)
        else:
            # If somehow we got no results for this model query, remove it from the dictionary and avoid iterating over it later
            results.pop(key)
            unique_model_versions.remove(key.split("_instances_")[1])

    # Now we gather the GPU metrics. The two most interesting ones for us are the DCGM_FI_PROF_PIPE_TENSOR_ACTIVE and 
    # DCGM_FI_PROF_DRAM_ACTIVE. The former measures how much of the compute resources (the Tensor Cores) are active, on average, in a time period
    # If the utilization is 50%, this could mean that the tensor cores for this GPU (slice) are 100% active for 50% of the time, 50% active for
    # 100% of the time, or any combination of activity_percent * time_active_percent that gives that product.
    gpu_queries = {"gpu_tensor_util_"+str(mg): "sum (avg_over_time(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{"+
                   "exported_container='triton',exported_namespace='triton',prometheus_replica='prometheus-k8s-0',"+
                   "device='"+gpu_inst.split("/")[0]+"',GPU_I_ID='"+gpu_inst.split("/")[1]+"',instance='"+gpu_inst.split("/")[2]+"'}["+step+"]))" for mg, gpu_inst in enumerate(unique_gpu_instances)}
    # An example of how additional labels can filter out non-matching queries, if we do 
    # DCGM_FI_PROF_DRAM_ACTIVE{exported_container='triton',exported_namespace='triton',prometheus_replica='prometheus-k8s-0',
    #                          device='nvidia2',GPU_I_ID='3',instance='110.4.29.45'}[120s]
    # We'll only get metrics from that specific device, if it has a running instance with that IP, and a running GPU instance matching it
    # In this case, for each timestep, it'll get a 'vector' of instantaenous measurements within 120s
    # The avg_over_time function then measures the average over time of that 'vector' and produces a scalar result
    # The scalar result may not be unique for a given timestamp, there can be other labels attached, and a final avg is taken over all
    # of those
    gpu_queries.update(
        {"gpu_dram_util_"+str(mg): "avg (avg_over_time(DCGM_FI_PROF_DRAM_ACTIVE{"+
         "exported_container='triton',exported_namespace='triton',prometheus_replica='prometheus-k8s-0',"+
        "device='"+gpu_inst.split("/")[0]+"',GPU_I_ID='"+gpu_inst.split("/")[1]+"',instance='"+gpu_inst.split("/")[2]+"'}["+step+"]))"
         for mg, gpu_inst in enumerate(unique_gpu_instances)})
    for key, query in gpu_queries.items():
        queries.append((key, query))
        results[key] = []
        for st, et in timestamp_tuples:
            test_inp = prom.custom_query_range(
                query=query,
                start_time=st,
                end_time=et,
                step=step
            )
            if len(test_inp) > 0:
                df = MetricRangeDataFrame(test_inp)
                results[key].append(df)
        if len(results[key]) > 0:
            results[key] = pd.concat(results[key], axis=0)
            #print(key)
        else:
            #print(f"results empty for {key}")
            results.pop(key)
            unique_gpu_instances.remove(key.split("_util_")[1])
    return results, queries, unique_model_versions, unique_gpu_instances