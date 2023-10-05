# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:26:41 2023

@author: athen
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import scipy.stats as stats
from datetime import datetime
from tabulate import tabulate
import seaborn as sns

#Loading Data


base = r'C:\Users\athen\OneDrive\Documents\GitHub\data-analysis-portfolio\RA_Project' #adjust your path!
teachers = pd.read_csv(os.path.join(base, 'Example_teacher_data_final.csv'))



#This is based off the analysis I did of an actual RCT for the TMW Center
#However of course this isn't the real data from the study because of confidentiality. 
#Instead this data was created in excel using packages which allowed me to draw 
#random data with different probability's for some of the binary demographic data 
#and following a normal distribution for the test score data. This included the 
#birthdates which were randomly drawn. The original study also included student 
#skills data collected at each of the three stages of the study for which I calculated 
#normalized scores; however to keep this sample from being too long. I chose not 
#to include this and some orginal components of the project here. 


#in the orignial dataproject more initial cleaning was nescessary including data 
#pivoting and merging multple datasets however I have simplified this to avoid
#this sample being too long. 

#first I'm making the treatment 1 and control 0, initially control was 2 and I 
#wanted use something more standard
teachers["admin condition_educator"] = teachers["admin condition_educator"].replace({2:0})



#Here I'm making a summed score for the main survey results, and averages for the 
#two behavioral surveys 

survey_list = ["survey_1", "survey_2", "survey_3"]
for survey in survey_list:
    question_numbers = list(range(1, 22))
    question_list = []
    behavior_weekly_list = []
    behavior_ordinal_list = []
    for question in question_numbers:
       survey_question_columns = survey+" question_" + str(question)
       question_list.append(survey_question_columns)
       teachers[survey + ' score total'] = teachers[question_list].sum(axis=1)
    question_numbers = list(range(1, 4)) 
    for question in question_numbers:
        survey_question_columns = survey+" behaviors_" + str(question)
        behavior_weekly_list.append(survey_question_columns)
        teachers[survey + ' behavior weekly total'] = (teachers[behavior_weekly_list].sum(axis=1))/4
    question_numbers = list(range(5, 8))
    for question in question_numbers:
        survey_question_columns = survey+" behaviors_" + str(question)
        behavior_ordinal_list.append(survey_question_columns)
        teachers[survey + ' behavior ordinal total'] = (teachers[behavior_ordinal_list].sum(axis=1))/4
        
#In summing the columns it summed NAs to 0s, so I'm turning them back into NAs
for survey in survey_list: 
    teachers[survey + ' behavior ordinal total'].replace(0, np.nan, inplace=True)
    teachers[survey + ' behavior weekly total'].replace(0, np.nan, inplace=True)
    teachers[ survey + ' score total'].replace(0, np.nan, inplace=True)


#Now I'm cleaning the demographic variables and creating dummy variables for ease 
#in analysis 



teachers['admin date_of_birth'] = pd.to_datetime(teachers['admin date_of_birth'])   
age_list_ed = []
for born in teachers['admin date_of_birth']:
    datetime_str = '09/01/22' #this was considered the official start of the study
    datetime_object = datetime.strptime(datetime_str, '%m/%d/%y')
    age = datetime_object.year - born.year - ((datetime_object.month, datetime_object.day) < (born.month, born.day))
    age_list_ed.append(age)
teachers['Age'] = age_list_ed




teachers.rename(columns={'admin ece_educator_race___1':'White', 'admin ece_female': 'Female', 
                         'admin ece_educator_race___2':'Black',  'admin ece_educator_race___3': 'Hispanic',
                         'survey_1 ece_educator_householdsize' : 'Household Size',
                         'survey_1 ece_educator_children': 'Number of Children',
                         'survey_1 ece_educator_foodstamps': 'Foodstamps',
                         'survey_1 ece_educator_wic': 'WIC',}, inplace=True)

teachers['admin ece_educator_yrs_ece'] = teachers['admin ece_educator_yrs_ece'].replace({1:'1 year or less in ECE', 2: '2 to 3 years in ECE', 3:'4 to 5 years in ECE', 
                                                          4:'6 to 10 years in ECE', 5: '10 years or more in ECE'})
teachers['survey_1 ece_educator_insurance'] = teachers['survey_1 ece_educator_insurance'].replace({1:'Insurance through employer or union', 
                                                          2: 'Insurance purchased directly through a company', 3:'Insurrance through Government assistance', 
                                                          4:'Other Insurance', 5: 'Uninsurred'})

teachers['admin ece_educator_edulevel'] = teachers['admin ece_educator_edulevel'].replace({6:'Associates Degree or some College Credits', 
                                                          7: 'Associates Degree or some College Credits', 
                                                          8:'BA degree',
                                                          9:'BA degree', 10: 'Masters degree'})


dummy_demographics = ['admin ece_educator_edulevel', 'admin ece_educator_yrs_ece', 'survey_1 ece_educator_insurance']
for column in dummy_demographics:
    col_df = pd.get_dummies(teachers[column])
    teachers = pd.concat([teachers,col_df],axis=1)


demographics = ['Associates Degree or some College Credits', 'BA degree',
'Masters degree', '1 year or less in ECE', '2 to 3 years in ECE', '4 to 5 years in ECE',
'6 to 10 years in ECE', '10 years or more in ECE', 'White', 'Black', 'Hispanic', 'Female', 'Age', 'Number of Children', 'Household Size', 
'Insurance through employer or union', 'Insurrance through Government assistance',
'Other Insurance', 'Insurance purchased directly through a company', 'Uninsurred', 'Foodstamps', 'WIC']


#Balance_checks 
# I used both a t-test and a Man Whitney Test to check for balance
# For the original study the N was even smaller and it was harder to 
# tell the distribution some of the data points came from, so I used the 
# Man-Whitney test as well which measures the distribution as opposed to the mean.


def balance_tables(col_list, df):
    df_treatment = df[df["admin condition_educator"] == 0]
    df_control = df[df["admin condition_educator"] == 1]
    dif_means =[]
    treatment_list = []
    control_list = []
    sample_size =[]
    p_MW = []
    p_ttest = []
    for name in col_list: 
        control = df_control[name].dropna() 
        treatment = df_treatment [name].dropna()
        n = control.count() + treatment.count()
        sample_size.append(n)
        dif_mean= round(treatment.mean() - control.mean(),2)
        treatment_list.append(round(treatment.mean(),2))
        control_list.append(round(control.mean(),2))
        dif_means.append(dif_mean)
        p_t_test = stats.ttest_ind(a=control, b=treatment, equal_var=True).pvalue
        p_t_test_stars = str(round(p_t_test,2))
        if p_t_test < 0.1 and p_t_test > 0.05:
            p_t_test_stars = p_t_test_stars + '*' 
        if p_t_test < 0.05 and p_t_test > 0.01:
            p_t_test_stars = p_t_test_stars + '**'
        if p_t_test < 0.01:
            p_t_test_stars = p_t_test_stars + '***'
        p_ttest.append(p_t_test_stars)
        U1, p = mannwhitneyu(control, treatment, method="exact")
        p_stars = str(round(p,2))
        if p < 0.1 and p > 0.05:
            p_stars = p_stars + '*' 
        if p < 0.05 and p > 0.01:
            p_stars = p_stars + '**' 
        if p < 0.01:
            p_stars = p_stars + '***' 
        p_MW.append(p_stars)
    Questions_p_values = pd.DataFrame({'Question': col_list, 'Treatment Mean': treatment_list, 'Control Mean': control_list, 'Difference in Means': dif_means, 'P T-Test': p_ttest, 'MW-P-value': p_MW, 'N': sample_size})
    return(Questions_p_values)
 
demographic_teachers = balance_tables(demographics,teachers)

fig, ax = plt.subplots()
t= ax.table(cellText=demographic_teachers.values, colLabels=demographic_teachers.columns, loc='center', cellLoc='left')
ax.axis('off')
t.auto_set_font_size(False) 
t.set_fontsize(8)
t.auto_set_column_width(col=list(range(len(demographic_teachers.columns)))) 
plt.show()

#There are only two variables that are significantly different at the 10% level 
#using a T-Test. These include having 1 year or less in early childhood education 
#and having other insurance. It isn't surprising that just by chance some of the 
#variables are significant at the 10% level given the number of variables. 
#These two variables I will include as covariates when running this as an OLS study

#Here I plotted the data to provide a sense of the relationship between control 
#and treatment and the main surveys, and the two sets of behavior based questions. 
#One of the behavior questions asked how many days a week do teachers engage in 
#key learning behaviors, the other survey asked from a scale to 1-4, 4 being always 
#and 1 being never do they encourage parents to engage in specific behavior  

def plot_maker(var, limL, limU, yname, labelset, main_name):
    days = [60,120,180]
    ed_control = teachers[teachers["admin condition_educator"] == 0]
    ed_treatment = teachers[teachers["admin condition_educator"] == 1]
    treated_mean = [ed_treatment["survey_1 " + var].mean(), ed_treatment["survey_2 " + var].mean(), ed_treatment["survey_3 " + var].mean()]
    control_mean = [ed_control["survey_1 " + var].mean(), ed_control["survey_2 " + var].mean(), ed_control["survey_3 " + var].mean()]
    treated_sd = [ed_treatment["survey_1 " + var].std(), ed_treatment["survey_2 " + var].std(), ed_treatment["survey_3 " + var].std()]
    control_sd = [ed_control["survey_1 " + var].std(), ed_control["survey_2 " + var].std(), ed_control["survey_3 " + var].std()]
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    fig.supylabel(yname)
    fig.suptitle(main_name)
    ax0.errorbar(days, treated_mean, yerr=treated_sd, fmt='-o', color='darkblue', ecolor='darkblue', label = "treatment", capsize= 5)
    ax0.set_title('Treatment')
    ax0.set_ylim ([limL, limU])
    for (daysi,meani) in zip(days, treated_mean):
        ax0.text(daysi + 7, meani -labelset, round(meani, 2), va='bottom', ha='center')
    ax1.errorbar(days, control_mean, yerr=control_sd, fmt='-o', color='darkred', ecolor='darkred', capsize= 5, label = "control") 
    ax1.set_title('Control')
    for (daysi,meani2) in zip(days, control_mean):
        ax1.text(daysi + 7, meani2 -labelset, round(meani2, 2), va='bottom', ha='center')
    ax1.set_ylim ([limL, limU]) 
    plt.xlabel("Days")
    plt.xlim([55,195])
    plt.show()
    

plot_maker("score total", 50, 80, "total score", 7, "Main Scores Control and Treatment Overtime")
plot_maker("behavior weekly total", 0, 7, "Number of Days per Week", 1, "On Avergage Number of Times Teachers Engaged in Key Behaviors")
plot_maker("behavior ordinal total", 0, 4, "Scale from Never (0) to Always (4)", 0.8, "Frequency Teachers Encouraged Parents to Engage in Key Behaviors")

#Its not surprising that there doesn't appear to be a significant relationship 
#between any of the variables because they were all just randomly generated here
#to demonstrate real work I did

#Running OLS 
#This function runs multiple regressions at the same time and extracts the key 
#information I needed in a table. If set to predict it creates a list of predicted 
#and actual y variables.


def model_maker_pandas(v, df, list_of_vals = None, predict = False):
    survey_list = ["survey_1", "survey_2", "survey_3"]
    for survey in survey_list:
        if survey == "survey_1" and list_of_vals is None:
            df_2 = pd.concat([df['survey_1 ' + v], df["admin condition_educator"]], axis=1).dropna()
            x = sm.add_constant(df_2["admin condition_educator"])
        elif survey == "survey_1" and list_of_vals is not None:
            list_ed = [df['survey_1 ' + v], df["admin condition_educator"]]
            list2 = ["admin condition_educator"]
            for i in list_of_vals:
                list_ed.append(df[i])
                list2.append(i)
            df_2= pd.concat(list_ed, axis=1).dropna()
            x = sm.add_constant(df_2[list2])  
        elif survey != "survey_1" and list_of_vals is None: 
             df_2= pd.concat([df[survey + ' ' + v], df['survey_1 ' + v], df["admin condition_educator"]], axis=1).dropna()
             x = sm.add_constant(df_2[["admin condition_educator", 'survey_1 ' + v]])
        else:
            list_ed = [df['survey_1 ' + v], df["admin condition_educator"], df[survey + ' ' + v]]
            list2 = ["admin condition_educator", 'survey_1 ' + v]
            for i in list_of_vals:
                list_ed.append(df[i])
                list2.append(i)
            df_2= pd.concat(list_ed, axis=1).dropna()
            x = sm.add_constant(df_2[list2])
        y = df_2[survey + ' ' + v]
        model = sm.OLS(y, x).fit()
        if predict is True:
            predictedValues = model.predict()
            predictedValues = pd.DataFrame(data=predictedValues)
            predictedValues['Y ' + survey +' ' +v] = y
            predictedValues = predictedValues.rename(columns={0: 'Predicted ' + survey + ' ' + v})
            if survey == 'survey_1':
                final = predictedValues
            else: 
                final = pd.concat([final,predictedValues], axis=1)
        else: 
            test = round(model.params,4)
            test_se = round(model.bse, 4) 
            test_pvalues = round(model.pvalues,3)
            test_pvalues.index = test_pvalues.index + " p-value"
            test_se.index = test_se.index + " SE"
            rsquared = round(model.rsquared,4)
            rsquared_adjusted = round(model.rsquared_adj,4)
            nrows = len(df_2.index)
            rsquared_series = pd.Series([rsquared,rsquared_adjusted, nrows])
            rsquared_series = rsquared_series.set_axis(["R-Squared", "Adj R-Squared", "N"])
            test = test.append(test_pvalues)
            test = test.append(test_se)
            test = test.append(rsquared_series)
            test_2 = test.to_frame()
            test_2= test_2.reset_index()
            test_2 = test_2.rename(columns={"index": "variable", 0: survey + ' ' + v})
            if survey == 'survey_1':
                final = test_2.sort_values('variable')
            else: 
                final = final.merge(test_2, how='outer', on='variable')
                final = final.sort_values('variable')
    return(final)
            



main_OLS_df = model_maker_pandas("score total", teachers)
behavior_ordinal_df = model_maker_pandas("behavior ordinal total", teachers)
behavior_weekly_df = model_maker_pandas("behavior weekly total", teachers)
print(tabulate(main_OLS_df, headers = 'keys', tablefmt = 'psql'))

#I also added covariates that weren't balanced
unbalanced_demographics = ['1 year or less in ECE', 'Other Insurance']
main_OLS_df_covariates = model_maker_pandas("score total", teachers, unbalanced_demographics)
print(tabulate(main_OLS_df_covariates, headers = 'keys', tablefmt = 'psql'))



#Visually Checking how well the model fits the data
def visual_check(v, teachers, survey, list_of_vals = None):    
    model_maker_pandas(v, teachers, list_of_vals = None, predict = False)
    predicted = model_maker_pandas(v, teachers, predict = True)
    predicted.columns.to_list()
    sns.lmplot( y = 'Predicted '+ survey + ' '+ v, x=  'Y ' + survey + ' '+ v, data=predicted, fit_reg=False)
    min_point = min(min(predicted['Predicted '+ survey + ' '+ v]), min(predicted['Y ' + survey + ' '+ v])) 
    max_point = max(max(predicted['Predicted '+ survey + ' '+ v]), max(predicted['Y ' + survey + ' '+ v])) 
    line_coords = np.arange(min_point, max_point)
    plt.plot(line_coords, line_coords, 
         color='darkred', linestyle='--')
    survey_name = survey.replace("_", " ").title()
    v = v.title()
    if list_of_vals is None:
        plt.title(survey_name + ' ' + v)
    else:
        plt.title( survey_name + ' ' + v + ' with Demographics')
    plt.xlabel ('Actual')
    plt.ylabel ('Predicted')
    plt.show()           


follow_up_surveys = ['survey_2', 'survey_3']
for survey in follow_up_surveys: 
    visual_check("score total", teachers, survey)

for survey in follow_up_surveys: 
    visual_check("score total", teachers, survey, unbalanced_demographics)

#However this doesn't end this stage of the teacher data for the project The full 
#intervention included the completion of 8 educational modules by teachers between 
#survey 1 and survey 2; however not all teachers completed the intervention Modules
# with a few only completing some of the modules.

#Here I created some plots demonstrating module completion. It shows that while 
#nobody in the control group got access to the modules some people in the treament 
#only partially recieved the intervention.

ax = teachers.groupby([ "admin condition_educator", "modules"]).size().unstack().plot(kind='bar', stacked=True, cmap= 'summer')
ax.set(xlabel =" Treatment Status", ylabel = "Number of Educators", title ='Completion of Intervention Modules')
legend_handles, _= ax.get_legend_handles_labels()
ax.legend( legend_handles, ["None",'One','Two', 'Three','All Eight'], title=' Intervention \n Modules Completed', bbox_to_anchor=(1,1))
ax.set(xticklabels=[" Control", " Treatment"])
ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
ax.set_title("Number of Intervention Modules Completed Amongst Educators")
y_offset = -2
for bar in ax.patches:
    height = bar.get_height()
    if height > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + bar.get_y() + y_offset,
            round(bar.get_height()),
            ha='center',
            color='black',
            #weight='bold',
            size=10
        )
        
        
#I also want to understand of those who completed the modules who took the surveys,
#because some people didn't take the survey  
    
treatment_data = teachers.loc[teachers["admin condition_educator"] == 1]    
treatment_data.loc[treatment_data["admin condition_educator"] == 1]
ed_data_plot = treatment_data[['modules', 'survey_1 score total',  'survey_2 score total', 'survey_3 score total', "admin condition_educator"]]
speak_list = ['survey_1 score total',  'survey_2 score total', 'survey_3 score total',"admin condition_educator"]
ed_data_plot = pd.melt(ed_data_plot, id_vars=['modules', "admin condition_educator"], value_vars=speak_list)
ed_data_plot= ed_data_plot.dropna()


ax2 = ed_data_plot.groupby([ 'variable', 'modules']).size().unstack().plot(kind='bar', stacked=True, cmap= 'tab20')
ax2.set(xlabel ="Completed the Surveys", ylabel = "Number of Treated Educators")
legend_handles, _= ax2.get_legend_handles_labels()
ax2.legend( legend_handles, ["None",'One','Two', 'Three','All Eight'], title=' Intervention \n Modules Complete', bbox_to_anchor=(1,1))
ax2.set(xticklabels=["Survey 1", "Survey 2", "Survey 3"])
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=0)
ax2.set_title('Educators Assigned the Treatment Group who Completed the Surveys \n a Breakdown of who Completed the Intervention Modules')
y_offset = -1.5
for bar in ax2.patches:
    height = bar.get_height()
    if height > 0:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + bar.get_y() + y_offset,
            round(bar.get_height()),
            ha='center',
            color='black',
            #weight='bold',
            size=10
        )


#To isolate the effects of completing the modules on test scores I ran
#a 2S2L Regression 

#First Stage

teachers2 = teachers[['modules', 'survey_1 score total',  'survey_2 score total', 'survey_3 score total', "admin condition_educator"]]
teachers2['const'] = 1

first_stage_model = sm.OLS(teachers2['modules'], teachers2[['const', 'admin condition_educator']], missing='drop').fit()
print(first_stage_model.summary())
teachers2['predicted'] = first_stage_model.predict()

#survey 2 
teachers2_survey2 = teachers2.dropna(subset=['survey_2 score total', 'survey_1 score total'])
model_ss_survey2 = sm.OLS(teachers2_survey2['survey_2 score total'], teachers2_survey2[['const', 'predicted', 'survey_1 score total']]).fit()
print(model_ss_survey2.summary())

#survey 3 
teachers2_survey2 = teachers2.dropna(subset=['survey_2 score total', 'survey_1 score total'])
model_ss_survey3 = sm.OLS(teachers2_survey2['survey_2 score total'], teachers2_survey2[['const', 'predicted', 'survey_1 score total']]).fit()
print(model_ss_survey3.summary())

#Of course there wasn't an effect of the modules in this example, because the data 
#was created using a random number generator; however there is a correlation between 
#the first survey results and the second and third because in creating the scores 
#I randomly pulled numbers from the first survey from a normal distribution and
#pulled for the second and third pulled a number from a normal distribution and
#then averaged the numbers from the first the survey and the new number pulled 
#to create a score from survey 2 and survey 3. 
