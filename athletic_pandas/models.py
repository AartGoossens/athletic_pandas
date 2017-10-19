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
    def _tau_w_balance(dcp):
        return 546*math.e**(-0.01*dcp) + 316

    @requires(columns=['power'], athlete=['cp', 'w_prime'])
    def w_balance(self, tau=None):
        w_balance = []
        work_below_cp = 0

        for i, power in enumerate(self.power):
            work_below_cp += min(self.athlete.cp, power)
            dcp = work_below_cp/(i+1)
            tau = self._tau_w_balance(dcp)

            w_exp_total = 0
            for v, power in enumerate(self.power[:i]):
                w_exp = max(0, power - self.athlete.cp)
                w_exp_total += w_exp*(math.e**(-(i-v)/tau))
            w_balance.append(self.athlete.w_prime - w_exp_total)

        return w_balance

    @requires(columns=['power'], athlete=['cp', 'w_prime'])
    def w_balance_3(self, tau=None):
        s_n = 0
        w_balance = []
        work_below_cp = 0
        
        for i, power in enumerate(self.power):
            work_below_cp += min(self.athlete.cp, power)
            dcp = work_below_cp/(i+1)
            tau = self._tau_w_balance(dcp)

            power_above_cp = power - self.athlete.cp
            w_exp = max(0, power_above_cp)

            s_n = s_n + w_exp*(math.e**(i/tau))
            w_balance.append(self.athlete.w_prime-s_n*math.e**(-i/tau))

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
