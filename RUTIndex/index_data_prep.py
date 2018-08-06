import pandas as pd

weeks_look_back=10

#file="C:\\Users\\csvai\\OneDrive\\Desktop\\Stephen\\RUTIndex\\data\\RUT_test.txt"
input_file="C:\\Users\\csvai\\OneDrive\\Desktop\\Stephen\\RUTIndex\\data\\all_indexes_weekly_30_yrs.csv"
output_file="C:\\Users\\csvai\\OneDrive\\Desktop\\Stephen\\RUTIndex\\data\\all_indexes_weekly_30_yrs_adj.csv"

def get_out_file(output_file):
    return output_file

def custom_round(x, base=0.05):
    return float(base * round(float(x)/base))


df = pd.read_csv(input_file)
total_num_weeks=df.shape[0]

print("dataframe shape",df.shape,df.shape[0])

print("DF High +/n",df.High)
current_week=df.loc[:(total_num_weeks-weeks_look_back-1),"High"].reset_index( )
future_week=df.loc[weeks_look_back:,"High"].reset_index()

print("Current WeeK",current_week[:])
print("Future Week",future_week)
df_multiplier=future_week.div(current_week)
#df_multiplier=df_multiplier.round(1)
df_multiplier=df_multiplier['High'].astype(float).apply(lambda x: custom_round(x, base=0.02))
df_multiplier=df_multiplier.round(3)  #remove weird significant digits

print(df_multiplier)
#print ("Multiplier", df_multiplier.High)



df_multiplier.to_csv(output_file ,index=False, header=True)