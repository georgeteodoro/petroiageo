from datetime import datetime

# expand
#     total-time
#     time-per-chunk-updated
#     n-chunks
#     n-chunks-updated
# f-sel (dist)
#     total-time
#     total-manager-time
#         request-time (from recv to send, separated by request type)
#         sync-time
#     time-per-worker
#         TD-create-time
#         request-job-comm-time (between manager.send and recv,
#                                both wait and msg-time)
#         insert-feature-time (add num features already in)
#         eval-bootstrap-time
#             time-per-chunk-per-well (individual lines)
#         sync-time
# predict
#     total-time
#     TD-create-time
#     insert-features-time
#     train-time-per-chunk
#     predict-time-per-chunk
#         prep-features-time
#         predict-time
#         update-time

base_str = '[PROFILING]'
expand_str = '[expand]'
fsel_str = '[f-sel]'
predict_str = '[predict]'


def it_str(it):
    return f'[it{it}]'


def timestamp(base_str):
    print(f'[TIMESTAMP][{base_str}] '\
          f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


# === Expand ==================================================================
def prof_expand_tot_time(it, time, config):
    print(f'{base_str}{expand_str}{it_str(it)}[total-time] {time}')


def prof_expand_chunks_time(it, time, config):
    print(f'{base_str}{expand_str}{it_str(it)}[ran-chunks-time] {time}')


def prof_expand_chunk_time(it, chunk, time, config):
    print(f'{base_str}{expand_str}{it_str(it)}[chunk{chunk}-time] {time}')


def prof_expand_chunks_ran(it, ran, total, config):
    print(f'{base_str}{expand_str}{it_str(it)}[chunks-ran] {ran} {total}')


# === Feature Selection - Manager =============================================
def prof_fsel_tot_time(it, time, config):
    print(f'{base_str}{fsel_str}{it_str(it)}[total-time] {time}')


def prof_fsel_manager_time(it, busy_time, total_time, config):
    print(f'{base_str}{fsel_str}{it_str(it)}[total-manager-time] '\
          f'{busy_time} {total_time}')


def prof_fsel_manager_req_time(it, f_it, time, config):
    print(f'{base_str}{fsel_str}{it_str(it)}[f_it{f_it}]'\
          f'[manager-req-time] {time}')


def prof_fsel_manager_sync_times(it, time, config):
    print(f'{base_str}{fsel_str}{it_str(it)}'\
          f'[manager-final-sync-time] {time}')


def prof_fsel_manager_sync_time(it, f_it, time, config):
    print(f'{base_str}{fsel_str}{it_str(it)}[f_it{f_it}]'\
          f'[manager-sync-time] {time}')


# === Feature Selection - Worker ==============================================
def prof_fsel_worker_times(it, worker, run_time, makespan, total_jobs, config):
    print(f'{base_str}{fsel_str}{it_str(it)}[w{worker}]'\
          f'[total-worker-time] {total_jobs} {run_time} {makespan}')


# def prof_fsel_worker_time(it, worker, f_it, time, total_jobs, config):
#     print(f'{base_str}{fsel_str}{it_str(it)}[w{worker}][f_it{f_it}]'\
#           f'[worker-time] {total_jobs} {time}')


def prof_fsel_worker_create_time(it, worker, time, config):
    print(f'{base_str}{fsel_str}{it_str(it)}[w{worker}]'\
          f'[worker-TD-create-time] {time}')


# Both waiting and msg receiving times
def prof_fsel_worker_comm_time(it, worker, time, config):
    print(f'{base_str}{fsel_str}{it_str(it)}[w{worker}]'\
          f'[worker-comm-time] {time}')


def prof_fsel_worker_insert_time(it, worker, f_it, time, config):
    print(f'{base_str}{fsel_str}{it_str(it)}[w{worker}][f_it{f_it}]'\
          f'[worker-insert-feature-time] {time}')


def prof_fsel_worker_eval_times(it, worker, f_it, time, config):
    print(f'{base_str}{fsel_str}{it_str(it)}[w{worker}][f_it{f_it}]'\
          f'[worker-eval-bootstrap-time] {time}')


# def prof_fsel_worker_eval_time(it, worker, f_it, well, chunk, time, config):
#     print(f'{base_str}{fsel_str}{it_str(it)}[w{worker}][f_it{f_it}]'\
#           f'[well{well}][chunk{chunk}][worker-eval-bootstrap-time] {time}')


def prof_fsel_worker_sync_time(it, worker, f_it, time, config):
    print(f'{base_str}{fsel_str}{it_str(it)}[w{worker}][f_it{f_it}]'\
          f'[worker-sync-time] {time}')


# === Predict =================================================================
def prof_predict_tot_time(it, time, config):
    print(f'{base_str}{predict_str}{it_str(it)} [predict-total-time] {time}')


def prof_predict_create_time(it, time, config):
    print(f'{base_str}{predict_str}{it_str(it)} '\
          f'[predict-TD-create-time] {time}')


def prof_predict_insert_time(it, n_feature, time, config):
    print(f'{base_str}{predict_str}{it_str(it)}[n_feature{n_feature}] '\
          f'[predict-insert-feature-time] {time}')


def prof_predict_train_times(it, time, config):
    print(f'{base_str}{predict_str}{it_str(it)} '\
          f'[predict-train-time] {time}')


# def prof_predict_train_time(it, chunk, time, config):
#     print(f'{base_str}{predict_str}{it_str(it)}[chunk{chunk}] '\
#           f'[predict-train-time] {time}')


def prof_predict_pred_times(it, time, config):
    print(f'{base_str}{predict_str}{it_str(it)} '\
          f'[predict-pred-time] {time}')


def prof_predict_pred_time(it, chunk, time, config):
    print(f'{base_str}{predict_str}{it_str(it)}[chunk{chunk}] '\
          f'[predict-pred-time] {time}')


def prof_predict_pred_insert_time(it, chunk, time, config):
    print(f'{base_str}{predict_str}{it_str(it)}[chunk{chunk}] '\
          f'[predict-pred-insert-time] {time}')


def prof_predict_pred_run_time(it, chunk, time, config):
    print(f'{base_str}{predict_str}{it_str(it)}[chunk{chunk}] '\
          f'[predict-pred-run-time] {time}')


def prof_predict_pred_update_time(it, chunk, time, config):
    print(f'{base_str}{predict_str}{it_str(it)}[chunk{chunk}] '\
          f'[predict-pred-update-time] {time}')
