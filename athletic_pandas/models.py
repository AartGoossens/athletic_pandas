import math

import numpy as np
import pandas as pd

from .base import BaseWorkoutDataFrame
from .helpers import requires

MEAN_MAX_POWER_INTERVALS = [
    1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90,\
    120, 180, 300, 600, 1200, 3600, 7200]


class WorkoutDataFrame(BaseWorkoutDataFrame):
    @requires(columns=['power'])
    def mean_max_power(self):
        mmp = pd.Series()
        length = len(self)

        for i in MEAN_MAX_POWER_INTERVALS:
            if i > length:
                break

            mmp = mmp.append(
                pd.Series(
                    [self.power.rolling(i).mean().max()],
                    [i]
                )
            )

        return mmp

    @requires(columns=['power'])
    def weighted_average_power(self):
        wap = self.power.rolling(30).mean().pow(4).mean()**(1/4)
        return wap

    @requires(columns=['power'], athlete=['weight'])
    def power_per_kg(self):
        ppkg = self.power / self.athlete.weight
        return ppkg

    @staticmethod
    def _tau_w_balance(power_above_cp):
        return 546*math.e**(0.01*power_above_cp) + 316

    @requires(columns=['power'], athlete=['cp', 'w_prime'])
    def w_balance(self, tau=None):
        instant_w_bal = self.athlete.w_prime
        w_balance = []
        w_balance.append(instant_w_bal)

        for power in self.power:
            power_above_cp = power - self.athlete.cp
            if power_above_cp >= 0:
                instant_w_bal = instant_w_bal - power_above_cp
            else:
                pass
            w_balance.append(instant_w_bal)

        return w_balance


class Athlete:
    def __init__(self, name=None, sex=None, weight=None, dob=None, ftp=None,
            cp=None, w_prime=None):
        self.name = name
        self.sex = sex
        self.weight = weight
        self.dob = dob
        self.ftp = ftp
        self.cp = cp
        self.w_prime = w_prime
