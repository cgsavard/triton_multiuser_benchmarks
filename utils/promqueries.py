from datetime import timedelta, datetime
import time
import json
import copy
import prometheus_api_client
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.metric_range_df import MetricRangeDataFrame
from prometheus_api_client.metric_snapshot_df import MetricSnapshotDataFrame
from prometheus_api_client.metrics_list import MetricsList
from prometheus_api_client.utils import parse_datetime
import pandas as pd
import numpy as np
from rich.progress import track

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

def prom_query_hash(query_result):
    """Return a string-key to hash the result of a query, based on the labels Prometheus attaches"""
    metric_dict = query_result['metric']
    key = ""
    for k, v in metric_dict.items():
        key += "($)" +  k + "::" + v
    return key

def prom_query_add(query_A, query_B):
    """Manually add two queries' results together, if they are compatible."""
    result = {}
    result['metric'] = copy.deepcopy(query_A['metric'])
    hash_A = prom_query_hash(query_A)
    hash_B = prom_query_hash(query_B)
    assert hash_A == hash_B, f"Incompatible metrics are being added: {query_A['metric']} |INCOMPATIBLE WITH| {query_B['metric']}"
    result['values'] = copy.deepcopy(query_A['values'])
    result['values'] += copy.deepcopy(query_B['values'])
    return result

def single_query_split(timestamp_tuples, 
                      query, 
                      step="120s", 
                      namespace='triton',
                      deduplicate=False,
                      dataframe_mode="individual", #"unified", "individual", "naive"
                      prom=None,
                      track=False):
    """Function for running a single query and returning the results as a list of individual dataframes (dataframe_mode='individual')
    or raw results (dataframe_mode='bypass')."""
    if prom is None:
        prom = PrometheusConnect(url="http://lsdataitb.fnal.gov:9009/prometheus", disable_ssl=True)
    results_dict = {}
    errors = []
    #print(f"Running Query: {query}")
    if track:
        iterable = track(timestamp_tuples, description = f"Retrieving")
    else:
        iterable = timestamp_tuples
    for st, et in iterable:
        test_inp = prom.custom_query_range(
            query=query,
            start_time=parse_datetime(st),
            end_time=parse_datetime(et),
            step=step
        )
        for query_result in test_inp:
            key = prom_query_hash(query_result)
            if key not in results_dict:
                results_dict[key] = query_result
            else:
                results_dict[key] = prom_query_add(results_dict[key], query_result)
            
    # Queries are converted to a pandas dataframe
    if dataframe_mode.lower() == "individual":
        results = []
        for key in results_dict:
            try:
                df = MetricRangeDataFrame(results_dict[key])
                if deduplicate:
                    df = df[~df.index.duplicated(keep='first')]
                results.append(df)
            except:
                errors.append({key: results_dict[key]})
    elif dataframe_mode.lower() == "bypass":
        results = list(results_dict.values())
    else:
        try:
            df = MetricRangeDataFrame(list(results_dict.values()))
            if deduplicate:
                df = df[~df.index.duplicated(keep='first')]
            results = [df]
        except:
            errors.append(results_dict)
            
    if len(results) > 0:
        temp = results
        return temp, errors
    else:
        return None, None

def get_all_queries_v3(timestamp_tuples, 
                       step="120s", 
                       granular_step=None, 
                       namespace='triton', 
                       deduplicate=True,
                       prom=PrometheusConnect(url="http://lsdataitb.fnal.gov:9009/prometheus", disable_ssl=True)
                      ):
    """Function for running most common prometheus queries for analyzing server behavior.
    step: time step for the query window, will be the index of resulting pandas dataframes. Should be minimally 4x as long as the most infrequent metric gathered, which at the EAF is 30s x 4 = 120s
    granular_step: parameter for auto-discovery of unique models and GPU instances in the active timeframe. Should subdivide the timestampe range into a minimum of subdivisions, i.e. use "1d" for a total range of "8d"
    namespace: Denotes which namespace to use for the metrics. At the LPC EAF, should be 'triton'
    deduplicate: Removes duplicate timestamps, which may be introduced with overlapping timestamp_tuples starts and stops"""
    
    rs = ""
    rsm = ""
    if isinstance(namespace, str):
        rs = "{namespace='"+namespace+"'}"
        rsm = ",namespace='"+namespace+"'"
    # A dictionary for our results
    results = {}
    errors = {}
    # Tuples of the queries we'll make, for debugging and info
    queries = []
    
    # Some queries are best created after understanding which unique models+version have been run in the triton servers
    # and which GPU instances have been active. These are then used to formulate model/version-specific and GPU-specific stats
    columns_step = granular_step if granular_step is not None else step
    unique_model_versions, inactive_model_versions = find_active_models(timestamp_tuples, step=columns_step, namespace=namespace, prom=prom)
    unique_gpu_instances = find_active_gpus(timestamp_tuples, step=columns_step, namespace=None, prom=prom) #different namespace entirely
    
    #Basic queries. Some of them are used as proxies to figure out the unqique queries to make later, like the "gpu_tensor_util" below
    for key, query in track({
        "num_instances": "count((sum by(pod) (delta(nv_inference_request_success"+rs+"["+step+"]))) > 0)",
        "inf_rate_net":"sum (rate(nv_inference_count"+rs+"["+step+"]))",
        "inf_reqs_net":"sum(rate(nv_inference_request_success"+rs+"["+step+"]))",
        "inf_req_dur_net": "avg (delta(nv_inference_request_duration_us"+rs+"["+step+"])/(0.001+delta(nv_inference_request_success"+rs+"["+step+"])))",
        "inf_que_dur_net": "avg (delta(nv_inference_queue_duration_us"+rs+"["+step+"])/(0.001+delta(nv_inference_request_success"+rs+"["+step+"])))",
        "inf_inp_dur_net": "avg (delta(nv_inference_compute_input_duration_us"+rs+"["+step+"])/(0.001+delta(nv_inference_request_success"+rs+"["+step+"])))",
        "inf_inf_dur_net": "avg (delta(nv_inference_compute_infer_duration_us"+rs+"["+step+"])/(0.001+delta(nv_inference_request_success"+rs+"["+step+"])))",
        "inf_out_dur_net": "avg (delta(nv_inference_compute_output_duration_us"+rs+"["+step+"])/(0.001+delta(nv_inference_request_success"+rs+"["+step+"])))",
        }.items(), description="Running General Queries"):
        # Log the queries, as they're easier to parse after being resolved fully
        queries.append((key, query))
        # Dataframes are concatenated together along the time (index value) axis
        results[key], errors[key] = single_query_split(timestamp_tuples, 
                                                      query, 
                                                      step=step, 
                                                      namespace=namespace,
                                                      deduplicate=deduplicate,
                                                      dataframe_mode="individual", #"unified", "individual", "naive"
                                                      prom=prom,
                                                      track=False,
                                                      )
        
    # Here we build the model-specific queries, getting both the number of unique number of Triton instances that served 
    # inference requests for this model, ad well as the inference rate and request durations of that model across 
    # all Triton instances active per time step
    if unique_model_versions is not None:
        model_queries = {"num_instances_"+model_version: "count((sum by(pod) (delta(nv_inference_request_success{model='"+
                         model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"]))) > 0)"
                         for model_version in unique_model_versions}
        model_queries.update(
            {"inf_rate_"+model_version: "sum (rate(nv_inference_count{model='"+
             model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"]))"
             for model_version in unique_model_versions})
        model_queries.update(
            {"avg_batchsize_"+model_version: "sum(delta(nv_inference_count{model='"+
             model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"]))/"+
             "sum(delta(nv_inference_exec_count{model='"+
             model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"]))"
             for model_version in unique_model_versions})
        model_queries.update(
            {"inf_reqs_"+model_version: "sum (rate(nv_inference_request_success{model='"+
             model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"]))"
             for model_version in unique_model_versions})
        model_queries.update(
            {"inf_req_dur_"+model_version: "avg (delta(nv_inference_request_duration_us{model='"+
             model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"])/"+
             "(0.001+delta(nv_inference_request_success{model='"+model_version.split("/")[0]+
             "',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"])))"
             for model_version in unique_model_versions})
        model_queries.update(
            {"inf_que_dur_"+model_version: "avg (delta(nv_inference_queue_duration_us{model='"+
             model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"])/"+
             "(0.001+delta(nv_inference_request_success{model='"+model_version.split("/")[0]+
             "',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"])))"
             for model_version in unique_model_versions})
        model_queries.update(
            {"inf_inp_dur_"+model_version: "avg (delta(nv_inference_compute_input_duration_us{model='"+
             model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"])/"+
             "(0.001+delta(nv_inference_request_success{model='"+model_version.split("/")[0]+
             "',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"])))"
             for model_version in unique_model_versions})
        model_queries.update(
            {"inf_inf_dur_"+model_version: "avg (delta(nv_inference_compute_infer_duration_us{model='"+
             model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"])/"+
             "(0.001+delta(nv_inference_request_success{model='"+model_version.split("/")[0]+
             "',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"])))"
             for model_version in unique_model_versions})
        model_queries.update(
            {"inf_out_dur_"+model_version: "avg (delta(nv_inference_compute_output_duration_us{model='"+
             model_version.split("/")[0]+"',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"])/"+
             "(0.001+delta(nv_inference_request_success{model='"+model_version.split("/")[0]+
             "',version='"+model_version.split("/")[1]+"'"+rsm+"}["+step+"])))"
             for model_version in unique_model_versions})
        for key, query in track(model_queries.items(), description="Running Model Queries"):
            queries.append((key, query))
            results[key], errors[key] = single_query_split(timestamp_tuples, 
                                                          query, 
                                                          step=step, 
                                                          namespace=namespace,
                                                          deduplicate=deduplicate,
                                                          dataframe_mode="individual", #"unified", "individual", "naive"
                                                          prom=prom,
                                                          track=False,
                                                          )
            if results[key] is None:
                # If somehow we got no results for this model query, remove it from the dictionary and avoid iterating over it later
                try:
                    results.pop(key)
                    unique_model_versions.remove(key.replace("inf_rate_", "").replace("num_instances_", ""))
                except:
                    pass

    # With the unique GPU instances known, we can create individual queries for each one's tensor and dram utilization
    # At the EAF, the device ('nvidiaX' where X is 0...4 for example), GPU instance ID (enumeration)
    # and the instance (IP address of host machine) are sufficient to make a unique identifier
    # The two most interesting metrics for us are the DCGM_FI_PROF_PIPE_TENSOR_ACTIVE and 
    # DCGM_FI_PROF_DRAM_ACTIVE. The former measures how much of the compute resources (the Tensor Cores) are active, on average, in a time period
    # If the utilization is 50%, this could mean that the tensor cores for this GPU are 100% active for 50% of the time, 50% active for
    # 100% of the time, or any combination of activity_percent * time_active_percent that gives that product.
    if unique_gpu_instances is not None:
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
        for key, query in track(gpu_queries.items(), description="Running GPU Queries"):
            queries.append((key, query))
            results[key], errors[key] = single_query_split(timestamp_tuples, 
                                                          query, 
                                                          step=step, 
                                                          namespace=namespace,
                                                          deduplicate=deduplicate,
                                                          dataframe_mode="individual", #"unified", "individual", "naive"
                                                          prom=prom,
                                                          track=False,
                                                          )
            if results[key] is None:
                #print(f"results empty for {key}")
                try:
                    results.pop(key)
                    unique_gpu_instances.remove(key.split("_util_")[1])
                except:
                    pass
    return results, errors, queries, unique_model_versions, unique_gpu_instances


def find_active_models(timestamp_tuples, step="120s", namespace=None, prom=None):
    """Function to identify the uniquely active model/version combinations in the timestamp range"""
    #st = timestamp_tuples[0][0]
    #et = timestamp_tuples[-1][1]
    #step = 
    results = single_query_split(
        [(timestamp_tuples[0][0], timestamp_tuples[-1][1])],
        "sum by(model, version) (rate(nv_inference_count["+step+"]))",
        step=step,
        namespace=namespace,
        dataframe_mode="bypass",
        deduplicate=False,
        prom=prom
    )[0] #Only need results
    active_models = []
    inactive_models = []
    for mv in results:
        vals = mv['values']
        total = sum([float(val[1]) for val in vals])
        if total > 0:
            active_models.append(mv['metric']['model'] + "/" + mv['metric']['version'])
            #print(total, mv['metric'])
        else:
            inactive_models.append(mv['metric']['model'] + "/" + mv['metric']['version'])
            #print("0 rate: ", mv['metric'])
    return active_models, inactive_models

def find_active_gpus(timestamp_tuples, step="120s", namespace=None, prom=None):
    """Function to identify the uniquely active GPU instances in the timestamp range"""
    results = single_query_split(
        [(timestamp_tuples[0][0], timestamp_tuples[-1][1])],
        "sum by(device, GPU_I_ID, instance) (avg_over_time (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{exported_container='triton',exported_namespace='triton',prometheus_replica='prometheus-k8s-0'}["+step+"]))",
        step=step,
        namespace=namespace,
        dataframe_mode="bypass",
        deduplicate=False,
        prom=prom
    )[0] #Only need results
    devices = []
    for mv in results:
        vals = mv['values']
        devices.append(mv['metric']['device'] + "/" + mv['metric']['GPU_I_ID'] + "/" + mv['metric']['instance'])
    return devices

def convert_results_to_df(results, step, unique_model_versions=None, unique_gpu_instances=None, add_model_stats=True, add_gpu_stats=False):
    """Convert results in the format returned by get_all_queries_v3 into a unified, aligned pandas dataframe."""
    # This iteratively walks through some of the dataframes that are compatible and aggregates results into a 
    # unified dataframe. In each dataframe, the join call, in combination with how='left', means that results are broadcast
    # and filled with NaN wherever results may be missing from the second of the two dataframes.
    # For this reason, the 'inf_rate_net' which should have a valid value for all timestamps is used as the base.
    ##idx = pd.period_range(min(df.date), max(df.date))
    ##...: results.reindex(idx, fill_value=0)
    min_dates = []
    max_dates = []
    for k, vlist in results.items():
        for v in vlist:
            min_dates.append(min(v.index.values))
            max_dates.append(max(v.index.values))
    min_date = min(min_dates)
    max_date = max(max_dates)
    new_index = pd.date_range(min_date, max_date, freq=step)
    ret = None
    concat_dict = {}
    for k, vlist in results.items():
        it = 0
        if len(vlist) > 1:
            print(f"Unable to add results column {k} due to multiple un-keyed results")
        else:
            # Make this an interable dict being returned from the split-query mode, then the above exception can be removed
            for v in vlist:
                it += 1
                try:
                    tmp = v.reindex(new_index, fill_value=0)
                except:
                    print(f"failure to reindex {k} {it} --> {v.columns}")
                    return new_index, v
            if ret is None:
                ret = tmp.rename(columns={"value": k})
            else:
                assert np.all(ret.index.values == tmp.index.values), "Mismatched Time Indices Detected"
                concat_dict[k] = tmp.value
                #ret.loc[:, k] = tmp.value
    ret = pd.concat((ret, pd.DataFrame(concat_dict)), axis='columns')
    return ret