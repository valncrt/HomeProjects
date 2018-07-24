import pandas as pd

weeks_look_back=10

#file="C:\\Users\\csvai\\OneDrive\\Desktop\\Stephen\\RUTIndex\\data\\RUT_test.txt"
file="C:\\Users\\csvai\\OneDrive\\Desktop\\Stephen\\RUTIndex\\data\\RUT_30_years.csv"
df = pd.read_csv(file)
total_num_weeks=df.shape[0]

print("dataframe shape",df.shape,df.shape[0])

print("DF High +/n",df.High)
current_week=df.loc[:(total_num_weeks-weeks_look_back-1),"High"].reset_index( )
future_week=df.loc[weeks_look_back:,"High"].reset_index()

print("Current WeeK",current_week[:])
print("Future Week",future_week)
df_multiplier=future_week.div(current_week)

print ("Multiplier", df_multiplier.High)


df_multiplier.to_csv("C:\\Users\\csvai\\OneDrive\\Desktop\\Stephen\\RUTIndex\\data\\RUT_30_years_high_multiplier.csv",columns=['High'] ,index=False, header=True)