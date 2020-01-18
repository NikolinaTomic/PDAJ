from datetime import datetime
import os
import socket
import subprocess
import time
import csv

import beam_integrals as bi
from beam_integrals.beam_types import BaseBeamType
from beam_integrals.integrals import BaseIntegral
from celery import chain, chord
from celery.exceptions import Reject
import numpy as np
import tables as tb

import pdaj as ebi
from ..app import app
from .worker import gen_simulation_model_params, solve

@app.task
def write_csv(results):
    with open("/results/dist60.csv", 'w') as csvfile:
        fieldnames = ['theta1_init', 'theta2_init', 'theta1', 'theta2']
        spamwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        spamwriter.writeheader()
        for result in results:
            spamwriter.writerow({'theta1_init': result[0], 'theta2_init': result[1],
                                 'theta1': result[2][-1], 'theta2': result[3][-1]})


## Seeding the computations
#docker exec -it pdaj_server_1 cat /var/log/supervisor/celery.log
@app.task
def seed_computations(ignore_result=True):
    if os.path.exists(get_experiment_status_filename('started')):
        raise Reject('Computations have already been seeded!')

    record_experiment_status.si('started').delay()
    chord(
        (solve.s(app.conf.DEFAULT_L1, app.conf.DEFAULT_L2, app.conf.DEFAULT_M1, app.conf.DEFAULT_M2, app.conf.DEFAULT_TMAX,
                 app.conf.DEFAULT_DT, theta1_init, theta2_init)
        for theta1_init, theta2_init in gen_simulation_model_params(app.conf.DEFAULT_RESOLUTION)),write_csv.s()).delay()

    record_experiment_status.si('completed').delay()

## Recording the experiment status

def get_experiment_status_filename(status):
    return os.path.join(app.conf.STATUS_DIR, status)

def get_experiment_status_time():
    """Get the current local date and time, in ISO 8601 format (microseconds and TZ removed)"""
    return datetime.now().replace(microsecond=0).isoformat()

@app.task
def record_experiment_status(status):
    with open(get_experiment_status_filename(status), 'w') as fp:
        fp.write(get_experiment_status_time() + '\n')
