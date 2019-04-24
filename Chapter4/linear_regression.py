import csv

epochs = 10000
learning_rate = 0.001
w = 0
b = 0

def calculate_loss(input_rows, w, b):
	loss = 0
	for row in input_rows:
		x = float(row[1]) # Advertisment Spending (The only input feature - x)
		y = float(row[2]) # Units Sold (y)
		y_hat = w * x + b
		loss += (y - y_hat)**2
	return loss/len(input_rows)

# Read data from csv file into input_rows
with open('Advertising_with_1_feature.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	input_rows = list(csv_reader)
csv_file.close()

# Perform epoch number of iterations through the entire input data
for ep in range(epochs):
	dl_dw = 0
	dl_db = 0
	# Loop through all the data inputs
	for row in input_rows:
		x = float(row[1]) # Advertisment Spending
		y = float(row[2]) # Units Sold (y)
		dl_dw += (- 2) * x * ( y - ( w * x + b ) )
		dl_db += (- 2) * ( y - ( w * x + b ) )
	dl_dw /= len(input_rows)
	dl_db /= len(input_rows)
	# Update weight and bias
	w -= learning_rate * dl_dw
	b -= learning_rate * dl_db

	#Print loss
	if ep % 400 == 0:
		print("Epoch:" , ep, "Loss:", calculate_loss(input_rows, w, b))