import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import linregress

def correlation_matrix(df):
	"""
	Ouptus a correlation matrix
	"""
	f = plt.figure(figsize=(11, 10))
	plt.matshow(df.corr(), fignum=f.number)
	plt.xticks(range(df.shape[1]), df.columns, fontsize=12, rotation=90)
	plt.yticks(range(df.shape[1]), df.columns, fontsize=12)
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=12)
	plt.title('Correlation Matrix', fontsize=16, y=-0.05)
	plt.savefig('correlation.png')
    # plt.show()

def plot_na(df):
    """
    Plots the perentage of NaN variables per column
    """
    plt.bar(range(int(len(df['columns']))), df['percent_nan'], color="blue")
    plt.xticks(fontsize=8)
    plt.title('Percent NaN by Column', fontsize=12)
    plt.ylabel('Percent NaN')
    plt.xlabel('Column Number')
    plt.savefig('na.png')
    # plt.show()

def scatter(df: pd.DataFrame, col_x: str, col_y:str, title:str):
    """
    Scatter plot of any 2 columns
    """
    plt.plot(df[col_x], df[col_y], 'b.', alpha=0.5, label='orignal data')
    
    slope, intercept, r_value, p_value, std_err = linregress(df[col_x], df[col_y])
    plt.plot(df[col_x], intercept + slope*df[col_x], 'r', label='fitted line')
    plt.title(title)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.legend()
    # plt.show()
    plt.savefig(title + '.png')

def main(infile = 'output.csv'):
    data = pd.read_csv(infile)
    na = pd.read_csv('nan.csv')
    
    correlation_matrix(data)

    plot_na(na)

    scatter(data, 'audience_average', 'audience_percent', 'Audience Average vs Audeince Percent')
    scatter(data, 'audience_average', 'critic_percent', 'Audience Average vs Critic Percent')
    scatter(data, 'audience_average', 'critic_average', 'Audience Average vs Critic Average')
    scatter(data, 'year', 'audience_average', 'Audience Average vs Year')
    



if __name__ == '__main__':
	main('output.csv')