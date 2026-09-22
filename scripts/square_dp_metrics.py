"""Task duration statistics; unsuccessful episodes are not completion times."""
import numpy as np


def distribution(values):
    x=np.asarray(values,dtype=float)
    return dict(count=len(x),mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                p95=float(np.percentile(x,95)) if len(x) else None)


def task_metrics(episodes):
    success=[e for e in episodes if e['success']]
    failed=[e for e in episodes if not e['success']]
    return dict(control_hz=20,success_count=len(success),failure_count=len(failed),
        success_steps=distribution([e['steps'] for e in success]),
        success_sim_seconds=distribution([e['steps']/20 for e in success]),
        success_execution_wall_seconds=distribution([e['execution_wall_seconds'] for e in success if 'execution_wall_seconds' in e]),
        failure_steps=distribution([e['steps'] for e in failed]),
        failure_sim_seconds=distribution([e['steps']/20 for e in failed]),
        all_episode_steps=distribution([e['steps'] for e in episodes]),
        definition='Success makespan: executed control steps to first success, seconds=steps/20. Failures are censored, never successful completion times. Execution wall includes inference, simulator, observation preparation and in-loop video recording; excludes reset, startup and final artifact writes.')
