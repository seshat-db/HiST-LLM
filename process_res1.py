#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_res1.py -- process NLP benchmark results, aggregate by NGA and time
"""

# libraries used
import pandas as pd
import numpy as np
from sklearn import metrics
import os
import pickle

# TODO: change to the correct directory with results !!
os.chdir('/home/dkondor/CSH/HoloSim/NLP')

# original results (submission)
# model_cols = ["gpt-4o", "gpt-3.5-turbo-0125","meta-llama/Llama-3-70b-chat-hf", "gpt-4-turbo-2024-04-09"]

# load all results
# all_res = pd.read_parquet('2024-06-04_fixed_dups_adjusted_regions.parquet')

# new results (rebuttal and camera-ready)
all_res = pd.read_parquet('results_2024-08-15-10_18_56.parquet')

model_names = {
    "gpt-4o-2024-05-13-answer": "GPT-4o",
    "gpt-3.5-turbo-0125-answer": "GPT-3.5-turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-answer": "Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo-answer": "Llama-3.1-70B",
    "meta-llama/Llama-3-70b-chat-hf-answer": "Llama-3-70B",
    "gpt-4-turbo-2024-04-09-answer": "GPT-4-turbo",
    "gemini-1.5-flash-answer": "Gemini-1.5-flash",
}
model_cols = list(model_names.keys())


###############################################################################
# 1. process data for the Appendix Figs. S2, S3 and S5: total counts and 
# average balanced accuracy scores per NGA over time


# 1.1. Appendix Fig S5.: counts of ground truth and model answers
tmp1 = model_cols.copy()
tmp1.append('A')

tmp3 = list()
for m in tmp1:
    tmp2 = all_res.groupby(m)['Q'].count().reset_index(name='cnt')
    tmp2.rename({m:'answer'}, axis=1, inplace=True)
    tmp2['model'] = m
    tmp3.append(tmp2)
answer_counts = pd.concat(tmp3)
answer_counts.to_csv('new_res_answer_counts.csv', index=False)


# 1.2. Appendix Figs. S2, S3:
# we aggregate by NGA and time, but we need to identify when multiple
# polities overlap


polities = all_res[['polity_old_id', 'start_year_int', 'end_year_int', 'nga']].drop_duplicates()
polities.sort_values(['nga', 'start_year_int'], inplace=True)
# sanity check
len(polities.polity_old_id.unique()) == polities.shape[0]
# OK, True

# get all overlaps
def get_overlaps(polities):
    curr = 0
    overlap_pairs = list()
    for i in range(1, polities.shape[0]):
        if polities.nga.iloc[i] != polities.nga.iloc[curr]:
            curr = i
            continue
        # here, current and i are in the same nga
        # we know that i starts after current
        while curr < i and polities.start_year_int.iloc[i] > polities.end_year_int.iloc[curr]:
            curr += 1
        if curr == i:
            continue
        # here curr and i overlap
        overlap_pairs.append((curr, i))
        # check the range [curr, i]
        for j in range(curr + 1, i):
            if polities.start_year_int.iloc[i] <= polities.end_year_int.iloc[j]:
                overlap_pairs.append((j,i))
    return overlap_pairs
    
overlap_pairs = get_overlaps(polities)
len(overlap_pairs)    
# 221

# take care of one year overlaps (polity end and start of next polity are the same year)
# by adjusting the end year
polities2 = polities.copy()
for (i,j) in overlap_pairs:
    if polities2.start_year_int.iloc[j] == polities2.end_year_int.iloc[i]:
        polities2.iloc[i, polities2.columns.get_loc('end_year_int')] -= 1

overlap_pairs2 = get_overlaps(polities2)
len(overlap_pairs2)
# 66, OK


# for the remaining overlap, we aggregate values over them
# for this, generate the list of "active" polities for each year
polity_ranges = pd.DataFrame({'nga': polities2.nga.iloc[i], 'id': polities2.polity_old_id.iloc[i],
                              'year': year} for i in range(polities2.shape[0])
         for year in range(polities2.start_year_int.iloc[i], polities2.end_year_int.iloc[i]+1))
polity_grp1 = polity_ranges.groupby(['nga', 'year'])['id'].apply(tuple).reset_index(name='id')
polity_grp2 = polity_grp1.groupby(['nga','id'])['year'].apply(lambda x : (min(x), max(x))).reset_index(name='year_range')
polity_grp2['start_year'] = list(x[0] for x in polity_grp2['year_range'])
polity_grp2['end_year'] = list(x[1] for x in polity_grp2['year_range'])

polity_grp2.to_parquet('new_res_polity_groups.parquet')

# calculate balanced accuracy for each time period based on the above groups
processed_res = list()
for i in range(polity_grp2.shape[0]):
    tmp2 = polity_grp2.iloc[i]
    tmp1 = all_res[all_res.polity_old_id.isin(tmp2.id)]
    processed_res += list(
        {'nga': tmp2.nga, 'start_year': tmp2.start_year, 'end_year': tmp2.end_year,
           'model': m, 'balanced_accuracy': metrics.balanced_accuracy_score(tmp1['A'], tmp1[m])}
        for m in model_cols
        )
processed_res = pd.DataFrame(processed_res)

# add regions for convenience
regions = {"Deccan": "Central and Southern Asia", "Kachi Plain": "Central and Southern Asia", "Garo Hills": "Central and Southern Asia", "Middle Ganga": "Central and Southern Asia", "Sogdiana": "Central and Southern Asia", "Susiana": "Central and Southern Asia", "Orkhon Valley": "Eastern and South-Eastern Asia", "Middle Yellow River Valley": "Eastern and South-Eastern Asia", "Kansai": "Eastern and South-Eastern Asia", "Southern China Hills": "Eastern and South-Eastern Asia", "Central Java": "Eastern and South-Eastern Asia", "Kapuasi Basin": "Eastern and South-Eastern Asia", "Cambodian Basin": "Eastern and South-Eastern Asia", "Lena River Valley": "Europe", "Iceland": "Europe", "Crete": "Europe", "Latium": "Europe", "Paris Basin": "Europe", "Finger Lakes": "Northern America", "Cahokia": "Northern America", "Basin of Mexico": "Latin America and the Caribbean", "Valley of Oaxaca": "Latin America and the Caribbean", "Lowland Andes": "Latin America and the Caribbean", "North Colombia": "Latin America and the Caribbean", "Cuzco": "Latin America and the Caribbean", "Upper Egypt": "Northern Africa and Western Asia", "Yemeni Coastal Plain": "Northern Africa and Western Asia", "Konya Plain": "Northern Africa and Western Asia", "Galilee": "Northern Africa and Western Asia", "Southern Mesopotamia": "Northern Africa and Western Asia", "Oro PNG": "Oceania", "Chuuk Islands": "Oceania", "Big Island Hawaii": "Oceania", "Ghanaian Coast": "Sub-Saharan Africa", "Niger Inland Delta": "Sub-Saharan Africa"}
processed_res['region'] = list(regions[x] for x in processed_res.nga)
processed_res.to_parquet('new_processed_res.parquet')


# also get the number of data points (Fig. S2)
n_data = list()
for i in range(polity_grp2.shape[0]):
    tmp2 = polity_grp2.iloc[i]
    tmp1 = all_res[all_res.polity_old_id.isin(tmp2.id)]
    n_data.append({'nga': tmp2.nga, 'start_year': tmp2.start_year,
                        'end_year': tmp2.end_year, 'N': tmp1.shape[0]})
n_data = pd.DataFrame(n_data)
n_data['region'] = list(regions[x] for x in n_data.nga)
n_data.to_parquet('new_n_data.parquet')



###############################################################################
# 2. main results: tables, with balanced accuracy calculated over different
# groups, and bootstrap confidence intervals for these

# 2.1. bootstrap methods
rng = np.random.default_rng()

def do_one_bootstrap(tmp2, nsamples = 1000, adj = False, base = dict()):
    tmp1 = list()
    N = tmp2.shape[0]
    for m in model_cols:
        ba_real = metrics.balanced_accuracy_score(tmp2['A'], tmp2[m], adjusted=adj)
        ba_sampled = np.zeros(nsamples)
        for i in range(nsamples):
            ix = rng.choice(N, N, True)
            ba_sampled[i] = metrics.balanced_accuracy_score(tmp2['A'].iloc[ix],
                                                tmp2[m].iloc[ix], adjusted=adj)
        pct = np.percentile(ba_sampled, [2.5, 97.5])
        tmpres = {'model': m, 'balanced_accuracy': ba_real,
            'stdev': np.std(ba_sampled), 'ba_95_lo': pct[0], 'ba_95_hi': pct[1]}
        tmp1.append(tmpres | base)
    return(tmp1)

def ba_sd_bootstrap(res, colname, nsamples = 1000, adj = False):
    tmp1 = list()
    cols = [None] if colname is None else res[colname].unique()
    for r in cols:
        tmp2 = res[res[colname] == r] if r is not None else res
        tmp1 += do_one_bootstrap(tmp2, nsamples, adj, {colname: r} if r is not None else dict())
        
    tmp1 = pd.DataFrame(tmp1)
    if colname is not None:
        tmp1 = tmp1.pivot(index=colname, columns='model',
                    values=['balanced_accuracy', 'stdev', 'ba_95_lo', 'ba_95_hi'])
    return(tmp1)


# 2.2. calculate main results, along with the count of data points in each case
# results by UN region (Table 3)
res1 = ba_sd_bootstrap(all_res, 'region')
cnts1 = all_res.groupby('region')['A'].count()
# results by variable category (Table 4)
res2 = ba_sd_bootstrap(all_res, 'root_cat')
cnts2 = all_res.groupby('root_cat')['A'].count()
# results by NGA (Table S5)
res3 = ba_sd_bootstrap(all_res, 'nga')
cnts3 = all_res.groupby('nga')['A'].count()


# 2.3. helpers to write out the results as Latex tables

model_srt = sorted(model_cols)

# final output: table with ranges + averages + numbers
# NeurIPS format uses toprule / midrule / bottomrule instead of hline !!
def write_latex_cmb(tmp1, varname, outf, cnts, model_srt, NeurIPSformat = True, mean_cnt = True, var_dict = None):
    toprule = 'toprule' if NeurIPSformat else 'hline'
    midrule = 'midrule' if NeurIPSformat else 'hline'
    bottomrule = 'bottomrule' if NeurIPSformat else 'hline'
    
    with open(outf, 'w') as of:
        # header
        n_models = len(model_srt)
        of.write('\\begin{tabular}{l')
        for i in range(n_models):
            of.write('r')
        of.write('rr}\n' if mean_cnt else '}\n')
        of.write('\\{} \n{} '.format(toprule, varname))
        for m in model_srt:
            of.write('& {} '.format(model_names[m]))
        if mean_cnt:
            of.write('& Mean & \\parbox{0.9cm}{Data points}')
        of.write('\\\\ \\{} \n'.format(midrule))
        for i in range(tmp1.shape[0]):
            name1 = tmp1.index[i]
            if var_dict:
                name1 = var_dict[name1]
            of.write('{} '.format(name1))
            jmax = np.argmax(tmp1['balanced_accuracy'].iloc[i])
            model_max = tmp1.balanced_accuracy.columns[jmax]
            for m in model_srt:
                ba = tmp1.balanced_accuracy[m].iloc[i]
                ba_lo = tmp1.ba_95_lo[m].iloc[i]
                ba_hi = tmp1.ba_95_hi[m].iloc[i]
                if m == model_max:
                    of.write('& {{\\bf {:.1f}}} '.format(100*ba))
                else:
                    of.write('& {:.1f} '.format(100*ba))
                of.write(' [{:.1f}, {:.1f}] '.format(100*ba_lo, 100*ba_hi))
            # also write the mean
            if mean_cnt:
                of.write('& {:.1f} & {} \\\\ \n'.format(100.0 * np.mean(tmp1.balanced_accuracy.iloc[i]),
                                                         int(cnts[cnts.index == tmp1.index[i]])))
            else:
                of.write('\\\\ \n')
        # write the mean and stdev for each column
        of.write('\\{} \n Mean '.format(midrule))
        mn1 = tmp1.balanced_accuracy.mean()
        model_max = mn1.index[mn1.argmax()]
        for m in model_srt:
            if m == model_max:
                of.write('& {{\\bf {:.1f}}} '.format(100 * mn1[m]))
            else:
                of.write('& {:.1f} '.format(100.0 * mn1[m]))
        if mean_cnt:
            of.write('& & ')
        of.write('\\\\ \n Std. deviation ')
        for m in model_srt:
            of.write('& {:.1f} '.format(100.0 * np.std(tmp1.balanced_accuracy[m])))
        if mean_cnt:
            of.write('& & ')
        of.write('\\\\ \n\\{} \n'.format(bottomrule))
        of.write('\\end{tabular}\n')


# 2.4. write out results (to be copied into the manuscript)
# write_latex_cmb(res1, 'Region', 'ba_bootstrap_cmb_region.out', cnts1)
# write_latex_cmb(res2, 'Variable category', 'ba_bootstrap_cmb_cat.out', cnts2)
# write_latex_cmb(res3, 'NGA', 'ba_bootstrap_cmb_nga.out', cnts3)


###############################################################
# new results (for rebuttal and camera-ready) -- TODO: change the last parameter to True if using the official NeurIPS format !
write_latex_cmb(res1, 'Region', 'new_res_ba_bootstrap_cmb_region.out', cnts1, model_srt, False)
write_latex_cmb(res2, 'Variable category', 'new_res_ba_bootstrap_cmb_cat.out', cnts2, model_srt, False)
write_latex_cmb(res3, 'NGA', 'new_res_ba_bootstrap_cmb_nga.out', cnts3, model_srt, False)



# 2.5. full dataset (Table 1)
tmp12 = ba_sd_bootstrap(all_res, None)
# alternative: only two answers possible (no inferred)
# C -> A and D -> C
def rep_inferred(x):
    y = x.copy()
    y[y == 'C'] = 'A'
    y[y == 'D'] = 'B'
    return y

all_res_noinf = all_res.copy()
all_res_noinf['A'] = rep_inferred(all_res_noinf['A'])
for m in model_cols:
    all_res_noinf[m] = rep_inferred(all_res_noinf[m])
tmp22 = ba_sd_bootstrap(all_res_noinf, None)

# full dataset, adjusted scores
tmp13 = ba_sd_bootstrap(all_res, None, adj = True)
tmp23 = ba_sd_bootstrap(all_res_noinf, None, adj = True)


# function to write out the main results (Table 1)
def write_main_tab_latex(res_ba4, res_ba2, res_adj4, res_adj2, outf, NeurIPSformat = True):
    toprule = 'toprule' if NeurIPSformat else 'hline'
    midrule = 'midrule' if NeurIPSformat else 'hline'
    bottomrule = 'bottomrule' if NeurIPSformat else 'hline'
    
    res_tmp = [res_ba4, res_ba2, res_adj4, res_adj2]
    model_max = list()
    for i in range(len(res_tmp)):
        jmax = np.argmax(res_tmp[i].balanced_accuracy)
        model_max.append(res_tmp[i].model[jmax])
    
    with open(outf, 'w') as of:
        of.write('\\begin{tabular}{l|rr|rr}\n')
        of.write('\\{}\n'.format(toprule))
        of.write('& \\multicolumn{2}{|c|}{Balanced accuracy} & \\multicolumn{2}{|c}{Adjusted balanced accuracy} \\\\ \n')
        of.write('Model & (4-choice) & (2-choice) & (4-choice) & (2-choice) \\\\')
        of.write('\\{}\n'.format(midrule))
        
        for m in model_srt:
            of.write('{} '.format(model_names[m]))
            for i in range(len(res_tmp)):
                of.write(' & ')
                j = np.where(res_tmp[i].model == m)[0][0]
                ba = res_tmp[i].balanced_accuracy[j]
                ba_lo = res_tmp[i].ba_95_lo[j]
                ba_hi = res_tmp[i].ba_95_hi[j]
                
                if m == model_max[i]:
                    of.write('{{\\bf {:.1f}}} '.format(100*ba))
                else:
                    of.write('{:.1f} '.format(100*ba))
                of.write(' [{:.1f}, {:.1f}] '.format(100*ba_lo, 100*ba_hi))
            of.write('\\\\\n')
        of.write('\\{}\n'.format(bottomrule))
        of.write('\\end{tabular}\n')
        
write_main_tab_latex(tmp12, tmp22, tmp13, tmp23, 'new_res_ba_main.out', False)


all_bootstrap_results = {
    'res1': res1,
    'res2': res2,
    'res3': res3,
    'tmp12': tmp12,
    'tmp13': tmp13,
    'tmp22': tmp22,
    'tmp23': tmp23,
    'cnts1': cnts1,
    'cnts2': cnts2,
    'cnts3': cnts3
    }


###############################################################################
# 3. group results by time (Table 2 and Fig. S4)
timebins = [(-10000, -8000), (-8000, -6000), (-6000, -4000),
 (-4000, -3500), (-3500, -3000), (-3000, -2500), (-2500, -2000), (-2000, -1500),
 (-1500, -1000), (-1000, -500), (-500, 0), (0, 500), (500, 1000), (1000, 1500),
 (1500, 2000)]

# polities are classified into a time bin based on the midpoint of their existence
polities2['midpoint'] = (polities2.start_year_int + polities2.end_year_int) / 2.0
# manually adjust JpJomo1 since it would fall out of our range
polities2.loc[polities2.polity_old_id == 'JpJomo1', 'midpoint'] = -9999

tmp3 = list()
for (tmin, tmax) in timebins:
    p1 = polities2[(polities2.midpoint < tmax) & (polities2.midpoint >= tmin)]
    tmp1 = all_res[all_res.polity_old_id.isin(p1.polity_old_id)]
    tmp3 += do_one_bootstrap(tmp1, base = {'tmin': tmin, 'tmax': tmax, 'cnt': tmp1.shape[0]})

res_timebins3 = pd.DataFrame(tmp3).pivot(index=['tmin', 'tmax', 'cnt'], columns='model',
            values=['balanced_accuracy', 'stdev', 'ba_95_lo', 'ba_95_hi'])



def print_date(of, t):
    if t == 0:
        of.write('0')
        return
    suffix = 'CE' if t > 0 else 'BCE'
    t = abs(t)
    if t >= 1000:
        t /= 1000
        if float.is_integer(t):
            t = int(t)
        of.write('{}k {}'.format(t, suffix))
    else:
        of.write('{} {}'.format(t, suffix))

def date_suffix(t):
    if t > 0:
        return 'CE'
    if t < 0:
        return 'BCE'
    return '' # 0

def process_date_number(t):
    if t == 0:
        return t
    t = abs(t)
    if t < 1000:
        return t
    t /= 1000
    if float.is_integer(t):
        t = int(t)
    return '{}k'.format(t)
    

def format_date_pair(t1, t2):
    s1 = date_suffix(t1)
    s2 = date_suffix(t2)
    t1 = process_date_number(t1)
    t2 = process_date_number(t2)
    if s1 == s2:
        # print only one suffix
        return '{} -- {} {}'.format(t1, t2, s1)
    else:
        # write both suffices
        return '{} {} -- {} {}'.format(t1, s1, t2, s2)

# output of the above
def write_latex_cmb_time(tmp1, varname, outf, model_srt, NeurIPSformat = True, mean_cnt = True):
    toprule = 'toprule' if NeurIPSformat else 'hline'
    midrule = 'midrule' if NeurIPSformat else 'hline'
    bottomrule = 'bottomrule' if NeurIPSformat else 'hline'
    
    with open(outf, 'w') as of:
        # header
        n_models = len(model_srt)
        of.write('\\begin{tabular}{l')
        for i in range(n_models):
            of.write('r')
        of.write('rr}\n' if mean_cnt else '}\n')
        of.write('\\{} \n{} '.format(toprule, varname))
        for m in model_srt:
            of.write('& {} '.format(model_names[m]))
        if mean_cnt:
            of.write('& Mean & \\parbox{0.9cm}{Data points}')
        of.write('\\\\ \\{} \n'.format(midrule))
        for i in range(tmp1.shape[0]):
            tmin = tmp1.index[i][0]
            tmax = tmp1.index[i][1]
            cnt = tmp1.index[i][2]
            of.write(format_date_pair(tmin, tmax))
            jmax = np.argmax(tmp1['balanced_accuracy'].iloc[i])
            model_max = tmp1.balanced_accuracy.columns[jmax]
            for m in model_srt:
                ba = tmp1.balanced_accuracy[m].iloc[i]
                ba_lo = tmp1.ba_95_lo[m].iloc[i]
                ba_hi = tmp1.ba_95_hi[m].iloc[i]
                if m == model_max:
                    of.write(' & {{\\bf {:.1f}}} '.format(100*ba))
                else:
                    of.write(' & {:.1f} '.format(100*ba))
                of.write(' [{:.1f}, {:.1f}] '.format(100*ba_lo, 100*ba_hi))
            # also write the mean and the total count of data points
            if mean_cnt:
                of.write(' & {:.1f} & {} \\\\ \n'.format(100.0 * np.mean(tmp1.balanced_accuracy.iloc[i]), cnt))
            else:
                of.write('\\\\ \n')
        # write the mean and stdev for each column
        of.write('\\{} \n Mean '.format(midrule))
        
        mn1 = tmp1.balanced_accuracy.mean()
        model_max = mn1.index[mn1.argmax()]
        for m in model_srt:
            if m == model_max:
                of.write('& {{\\bf {:.1f}}} '.format(100 * mn1[m]))
            else:
                of.write('& {:.1f} '.format(100.0 * mn1[m]))
        if mean_cnt:
            of.write('& & ')
        of.write('\\\\ \n Std. deviation ')
        for m in model_srt:
            of.write('& {:.1f} '.format(100.0 * np.std(tmp1.balanced_accuracy[m])))
        if mean_cnt:
            of.write('& & ')
        of.write('\\\\ \n\\{} \n'.format(bottomrule))
        of.write('\\end{tabular}\n')

write_latex_cmb_time(res_timebins3, 'Time range', 'new_res_ba_bootstrap_time.out', model_srt, False)


all_bootstrap_results['res_timebins3'] = res_timebins3
with open('new_bootstrap_res.pkl', 'wb') as pkout:
    pickle.dump(all_bootstrap_results, pkout)


# write out results in a csv format for Fig. S4
tmp12 = res_timebins3['balanced_accuracy'].reset_index()
tmp12.to_csv('new_ba_time.csv', index=False)



######################################################################
# read previously saved results
with open('new_bootstrap_res.pkl', 'rb') as pkin:
    all_bootstrap_results = pickle.load(pkin)
# re-create all variables
globals().update(all_bootstrap_results)


######################################################################
# Camera-ready paper: 4 models go in main text, rest + mean and count in SI


# shorten the row names
region_names = {
    'Central and Southern Asia': 'Central, S Asia',
    'Eastern and South-Eastern Asia': 'E, SE Asia',
    'Europe': 'Europe',
    'Latin America and the Caribbean': 'Latin America',
    'Northern Africa and Western Asia': 'N Africa, W Asia',
    'Northern America': 'N America',
    'Oceania': 'Oceania',
    'Sub-Saharan Africa': '\\parbox{1.8cm}{Sub-Saharan\\\\Africa}'
}

cat_names = {
    'Cults and Rituals': 'Cults and Rituals',
    'Economy variables (polity-level)': 'Economy',
    'Equity': 'Discrimination',
    'Institutional Variables': 'Institutions',
    'Legal System': 'Legal System',
    'Religion and Normative Ideology': '\\parbox{2.8cm}{Religion and\\\\Normative Ideology}',
    'Social Complexity variables': 'Social Complexity',
    'Social Mobility': 'Social Mobility',
    'Warfare variables': 'Warfare variables',
    'Well-Being': 'Well-Being'
}


# main text: GPT-4o, Llama-3.1-70B, GPT-4-Turbo, Gemini
model_main = sorted([model_cols[0]] + [model_cols[3]] + model_cols[5:7])
# Appendix: GPT-3.5-Turbo, Llama-3.1-8B, Llama-3-70B
model_si = sorted(model_cols[1:3] + [model_cols[4]])

write_latex_cmb(res1, 'Region', 'new_res_ba_bootstrap_cmb_region_main.out', cnts1,
                model_main, True, False, region_names)
write_latex_cmb(res1, 'Region', 'new_res_ba_bootstrap_cmb_region_si.out', cnts1,
                model_si, True, True, region_names)

write_latex_cmb(res2, 'Variable category', 'new_res_ba_bootstrap_cmb_cat_main.out', cnts2,
                model_main, True, False, cat_names)
write_latex_cmb(res2, 'Variable category', 'new_res_ba_bootstrap_cmb_cat_si.out', cnts2,
                model_si, True, True, cat_names)

# Note: both of these go in the SI
write_latex_cmb(res3, 'NGA', 'new_res_ba_bootstrap_cmb_nga_main.out', cnts3,
                model_main, True, False)
write_latex_cmb(res3, 'NGA', 'new_res_ba_bootstrap_cmb_nga_si.out', cnts3,
                model_si, True, True)

write_latex_cmb_time(res_timebins3, 'Time range', 'new_res_ba_bootstrap_time_main.out',
                     model_main, True, False)
write_latex_cmb_time(res_timebins3, 'Time range', 'new_res_ba_bootstrap_time_si.out',
                     model_si, True, True)




