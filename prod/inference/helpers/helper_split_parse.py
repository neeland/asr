import pandas as pd

if __name__ == "__main__":

    csv_out_6000 = pd.read_csv("csv_out_6000.csv")
    csv_out_0_3000 =csv_out_6000[0:3000]
    csv_out_3000_6000 = csv_out_6000[3000:]
    csv_out_0_3000.to_csv(f'split/csv_out_0_3000.csv')
    csv_out_3000_6000.to_csv(f'split/csv_out_3000_6000.csv')


