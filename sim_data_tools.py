# Class for simulating spike trains based on different probability distributions

import math
import numpy as np
import pandas as pd
import random
import vonMises.vonMises as VM

from process_lfps import butter_bandpass, butter_bandpass_filter
from process_trial_data import find_true_trial_end_samples


# def sim_binom_spktrn(probs):
#     """Simulate binary spike train from binomial probability distribution
#     Param:
#     probs: list of probabilities that unfold over time"""
#
#     binary_spktrn = []
#     for i in range(len(probs)):
#
#         binary_spktrn.append(np.random.binomial(1,probs[i]))
#
#     return binary_spktrn

# def sim_binom_spktrn(probs,is_uniform,with_refractory,frate_target):
#     """Simulate binary spike train from binomial probability distribution
#     Param:
#     probs: list of probabilities that unfold over time
#     with_refractory: boolean, 1 to add refractory period
#     frate_target: scalar, firing rate target, in Hz"""
#     binary_spktrn = []
#     if frate_target == []:
#
#         if with_refractory == 0:
#
#             for i in range(len(probs)):
#
#                 binary_spktrn.append(np.random.binomial(1,probs[i]))
#
#         else:
#
#             for i in range(len(probs)):
#
#                 tmp = np.random.binomial(1,probs[i])
#
#                 binary_spktrn.append(tmp)
#
#                 if (tmp == 1) & (i != len(probs)-1):
#                     probs[i+1] = 0.00001
#                     if (tmp == 1) & (i+1 != len(probs)-1):
#                         probs[i+2] = (0.00001)*.375 + 0.00001
#                         if (tmp == 1) & (i+2 != len(probs)-1):
#                             probs[i+3] = (0.00001)*.5 + 0.00001
#     else:
#
#         if is_uniform == 0:
#             rescale_factor = (frate_target/1000)*6
#             probs = probs*rescale_factor
#
#
#         if with_refractory == 0:
#
#             for i in range(len(probs)):
#
#                 binary_spktrn.append(np.random.binomial(1,probs[i]))
#
#         else:
#
#             for i in range(len(probs)):
#
#                 tmp = np.random.binomial(1,probs[i])
#
#                 binary_spktrn.append(tmp)
#
#                 if (tmp == 1) & (i != len(probs)-1):
#                     probs[i+1] = 0.00001
#                     if (tmp == 1) & (i+1 != len(probs)-1):
#                         probs[i+2] = (0.00001)*.375 + 0.00001
#                         if (tmp == 1) & (i+2 != len(probs)-1):
#                             probs[i+3] = (0.00001)*.5 + 0.00001
#
#
#     return binary_spktrn, probs
def sim_binom_spktrn(probs,is_uniform,with_refractory,frate_target):
    """Simulate binary spike train from binomial probability distribution
    Param:
    probs: list of probabilities that unfold over time
    with_refractory: boolean, 1 to add refractory period
    frate_target: scalar, firing rate target, in Hz"""
    binary_spktrn = []
    if frate_target == []:

        if with_refractory == 0:

            for i in range(len(probs)):

                binary_spktrn.append(np.random.binomial(1,probs[i]))

        else:

            for i in range(len(probs)):

                tmp = np.random.binomial(1,probs[i])

                binary_spktrn.append(tmp)

                if (tmp == 1) & (i != len(probs)-1):
                    probs[i+1] = 0.00001
                    if (tmp == 1) & (i+1 != len(probs)-1):
                        probs[i+2] = (0.00001)*.375 + 0.00001
                        if (tmp == 1) & (i+2 != len(probs)-1):
                            probs[i+3] = (0.00001)*.5 + 0.00001
    else:

        if is_uniform == 0:
            rescale_factor = (frate_target/1000)*6
            probs = probs*rescale_factor


        if with_refractory == 0:

            for i in range(len(probs)):

                binary_spktrn.append(np.random.binomial(1,probs[i]))

        else:

            for i in range(len(probs)):

                tmp = np.random.binomial(1,probs[i])

                binary_spktrn.append(tmp)

                if (tmp == 1) & (i != len(probs)-1):
                    probs[i+1] = 0.00001
                    if (tmp == 1) & (i+1 != len(probs)-1):
                        probs[i+2] = (0.00001)*.375 + 0.00001
                        if (tmp == 1) & (i+2 != len(probs)-1):
                            probs[i+3] = (0.00001)*.5 + 0.00001


    return binary_spktrn, probs


def make_phase_timeseries(freq,duration,fs):

    freq_mHz = freq/fs

    n_cycles_timeseries = freq_mHz * duration

    one_cycle = np.linspace(np.pi,-np.pi,int(1/freq_mHz))
    p = np.tile(one_cycle,int(n_cycles_timeseries))

    if len(p) < duration:
        p = np.hstack((p,one_cycle))
        p = p[:duration]

    phases = p

    return phases


# def find_true_trial_end_samples(n_trials,true_len_trial):
#     """Identifies and saves the time samples immediately following the last
#     time sample in a trial--immediately after so that you can slice with it
#     and know that you will select all samples up to but not including, the time
#     you have input"""
#
#     true_trial_ends_not_inclusive = []
#     for i in range(1,n_trials+1):
#         true_trial_ends_not_inclusive.append((true_len_trial-1)*i)
#
#     return true_trial_ends_not_inclusive

#source for function below: https://www.dsprelated.com/showarticle/908.php
#Allen Downey, Olin College, Needham, MA
def voss(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.

    nrows: number of values to generate
    rcols: number of random sources to add

    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values

def make_lfp_from_phases(phase_timeseries):
    """phase_timeseries: list of phase values from -pi to pi"""

    sineseries = []
    for i in range(len(phase_timeseries)):
        sineseries.append(math.sin(phase_timeseries[i]))

    pinknoiseseries = voss(len(sineseries))

    signal = (sineseries+pinknoiseseries)
    signal = signal-np.mean(signal)

    return signal

class Neuron:

    def __init__(self, avg_frate, n_trials, has_rhythm, has_refractory, freq_seq, mu, kappa, fs, len_trial):
        """freq_seq: list of frequency values that interneuron prefers and switches to within
        a single trial (e.g. interneuron goes from 8 Hz to 30 Hz to 8 Hz within single trial)
        has_refractory: boolean, 1: has a refractory period, 0: does not"""

        self.avg_frate = avg_frate
        self.n_trials = n_trials
        self.has_rhythm = has_rhythm
        self.has_refractory = has_refractory
        self.freq_seq = freq_seq
        self.mu = mu
        self.kappa = kappa
        self.fs = fs
        self.len_trial = len_trial

    def make_singletrial_phase(self):
        """mu: list of phase preferences for each frequency listed in freq_seq"""

        len_trial = self.len_trial
        freq_mHz = [i/self.fs for i in self.freq_seq]

        n_cycles_timeseries = [i * len_trial/len(self.freq_seq) for i in freq_mHz]

        p=[]
        mu_list=[]
        kappa_list=[]
        for i,val in enumerate(freq_mHz):
            tmp = np.linspace(np.pi,-np.pi,int(1/val))
            p.append(np.tile(tmp,int(n_cycles_timeseries[i])))

            mu_list.append(np.repeat(self.mu[i],len(p[i])))
            kappa_list.append(np.repeat(self.kappa[i],len(p[i])))

        self.phase_single_trial = [item for sublist in p for item in sublist]
        self.mus_single_trial = [item for sublist in mu_list for item in sublist]
        self.kappas_single_trial = [item for sublist in kappa_list for item in sublist]

        self.true_len_trial = len(self.phase_single_trial)

        if self.true_len_trial < self.len_trial:

            #find the number of remaining time samples
            remaining_samples = self.len_trial - self.true_len_trial

            #take the last "val" in the frequency sequence to finish out
            #any remaining time samples to meet the desired trial length
            last_freq = freq_mHz[-1]

            tmp = np.linspace(np.pi,-np.pi,int(1/last_freq))

            #out of this cycle, keep only those timesamples that allow you to meet
            #the target trial length
            remaining_phases = tmp[:remaining_samples]

            #now just append these remaining phases at the end of phase time series
            self.phase_single_trial = np.hstack((self.phase_single_trial,remaining_phases))


            #make sure that the list of mus and kappas is also adjusted to match
            #the length of the phase time series
            mu_list = np.repeat(self.mu[-1],remaining_samples)
            self.mus_single_trial = np.hstack((self.mus_single_trial,mu_list))

            kappa_list = np.repeat(self.kappa[-1],remaining_samples)
            self.kappas_single_trial = np.hstack((self.kappas_single_trial,kappa_list))


        if self.true_len_trial > self.len_trial:

            #this problem is simpler: just shave off anything after the last
            #desired time sample
            self.phase_single_trial = self.phase_single_trial[:self.len_trial]
            self.mus_single_trial = self.mus_single_trial[:self.len_trial]
            self.kappas_single_trial = self.kappa_list[:self.len_trial]


        return self.phase_single_trial, self.mus_single_trial,self.kappas_single_trial, self.true_len_trial

    def make_multitrial_phase(self):
        """"""

        d = {'phases': np.tile(self.phase_single_trial,self.n_trials),
            'mus': np.tile(self.mus_single_trial,self.n_trials),
            'kappas': np.tile(self.kappas_single_trial,self.n_trials)}

        self.df = pd.DataFrame(d)

        return self.df

    def make_spike_probs(self):
        """"""

        self.spike_probs=[]

        if self.has_rhythm == 1:
            for i,val in enumerate(self.df['mus'].values):

                p = VM.dvonmises([self.df['phases'].values[i]],
                                             val,
                                             self.df['kappas'].values[i])
                self.spike_probs.append(p[0])

        else:
            for i,val in enumerate(self.df['mus'].values):

                p = self.avg_frate/self.fs

                self.spike_probs.append(p)


        self.df['spike_probs'] = self.spike_probs # easier to work with this data type than just the spike_probs by self


        #sample spikes from binomial distribution
        if self.has_rhythm == 1:
            is_uniform = 0
        else:
            is_uniform = 1

        self.spikes, self.spike_probs = sim_binom_spktrn(self.df['spike_probs'].values,is_uniform,self.has_refractory,self.avg_frate)


        self.df['spike_probs'] = self.spike_probs
        self.df['spikes'] = self.spikes

        return self.df

    def make_lfp_from_phases(self):

        phase_list = self.df['phases'].to_list()

        sineseries = []
        for i in range(len(phase_list)):
            sineseries.append(math.sin(phase_list[i]))

        pinknoiseseries = voss(len(sineseries))

        signal = (sineseries+pinknoiseseries)
        signal = signal-np.mean(signal)

        self.df['lfp'] = signal

        return self.df

    def label_trials(self):
        """"""

        true_trial_bins = find_true_trial_end_samples(self.n_trials,
                                                     self.len_trial)
        true_trial_bins.insert(0,0)
        labels = np.arange(len(true_trial_bins)-1)

        self.df['trial_labels'] = pd.cut(self.df.index.values,
                                         bins=true_trial_bins,
                                         labels=labels,
                                         include_lowest=True)


        return self.df

    def filter_lfp_phase_estimates(self,freq_range,new_col_name):

        _, phase, _, _= butter_bandpass_filter(self.df['lfp'].values, freq_range[0], freq_range[1], self.fs, order=3)

        col_name = new_col_name + '_' + str(freq_range[0]) + '_' + str(freq_range[1]) + '_phases'

        self.df[col_name] = phase

        return self.df
