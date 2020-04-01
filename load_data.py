import pandas as pd

lotto = 'a94c65f6-7123-11ea-835d-1868b10e31b6'
path = './Source/'
file_extension = '.xlsx'

file = path + lotto + file_extension

df = pd.read_excel(file)
data_lotto = df[6:]
print(data_lotto[:1])
# u = np.array(df['u'])
# print(type(u))
# y = np.array(df['y'])    
