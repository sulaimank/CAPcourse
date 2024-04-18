import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Convert the provided data into a pandas DataFrame
data = {
    "Car_id": ["C_CND_000001", "C_CND_000002", "C_CND_000003", "C_CND_000004", "C_CND_000005", "C_CND_000006", "C_CND_000007", "C_CND_000008", "C_CND_000009", "C_CND_000010", "C_CND_000011", "C_CND_000012", "C_CND_000013", "C_CND_000014", "C_CND_000015", "C_CND_000016", "C_CND_000017", "C_CND_000018", "C_CND_000019", "C_CND_000020", "C_CND_000021", "C_CND_000022"],
    "Date": ["1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022", "1/2/2022"],
    "Customer Name": ["Geraldine", "Gia", "Gianna", "Giselle", "Grace", "Guadalupe", "Hailey", "Graham", "Naomi", "Grayson", "Gregory", "Amar'E", "Griffin", "Harrison", "Zainab", "Zara", "Zoe", "Zoey", "Aaliyah", "Abigail", "Adrianna", "Joshua"],
    "Gender": ["Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Female", "Male", "Male", "Male", "Male", "Male", "Male", "Female", "Female", "Male", "Male", "Male", "Male"],
    "Annual Income": [13500, 1480000, 1035000, 13500, 1465000, 850000, 1600000, 13500, 815000, 13500, 13500, 13500, 885000, 13500, 722000, 746000, 535000, 570000, 685000, 455000, 13500, 2500000],
    "Dealer Name": ["Buddy Storbeck's Diesel Service Inc", "C & M Motors Inc", "Capitol KIA", "Chrysler of Tri-Cities", "Chrysler Plymouth", "Classic Chevy", "Clay Johnson Auto Sales", "U-Haul CO", "Rabun Used Car Sales", "Rabun Used Car Sales", "Race Car Help", "Race Car Help", "Saab-Belle Dodge", "Scrivener Performance Engineering", "Buddy Storbeck's Diesel Service Inc", "C & M Motors Inc", "Capitol KIA", "Chrysler of Tri-Cities", "Chrysler Plymouth", "Classic Chevy", "Clay Johnson Auto Sales", "Classic Chevy"],
    "Company": ["Ford", "Dodge", "Cadillac", "Toyota", "Acura", "Mitsubishi", "Toyota", "Mitsubishi", "Chevrolet", "Ford", "Acura", "Nissan", "Mercury", "BMW", "Chrysler", "Subaru", "Hyundai", "Cadillac", "Toyota", "Honda", "Toyota", "Infiniti"],
    "Model": ["Expedition", "Durango", "Eldorado", "Celica", "TL", "Diamante", "Corolla", "Galant", "Malibu", "Escort", "RL", "Pathfinder", "Grand Marquis", "323i", "Sebring Coupe", "Forester", "Accent", "Eldorado", "Land Cruiser", "Accord", "4Runner", "I30"],
    "Engine": ["Double Overhead Camshaft", "Double Overhead Camshaft", "Overhead Camshaft", "Overhead Camshaft", "Double Overhead Camshaft", "Overhead Camshaft", "Overhead Camshaft", "Double Overhead Camshaft", "Overhead Camshaft", "Double Overhead Camshaft", "Overhead Camshaft", "Double Overhead Camshaft", "Double Overhead Camshaft", "Double Overhead Camshaft", "Overhead Camshaft", "Overhead Camshaft", "Overhead Camshaft", "Double Overhead Camshaft", "Double Overhead Camshaft", "Double Overhead Camshaft", "Overhead Camshaft", "Double Overhead Camshaft"],
    "Transmission": ["Auto", "Auto", "Manual", "Manual", "Auto", "Manual", "Manual", "Auto", "Manual", "Auto", "Manual", "Auto", "Auto", "Auto", "Manual", "Manual", "Manual", "Auto", "Auto", "Auto", "Manual", "Auto"],
    "Color": ["Black", "Black", "Red", "Pale White", "Red", "Pale White", "Pale White", "Pale White", "Pale White", "Pale White", "Pale White", "Pale White", "Black", "Pale White", "Pale White", "Pale White", "Black", "Pale White", "Pale White", "Pale White", "Black", "Black"],
    "Price ($)": [26000, 19000, 31500, 14000, 24500, 12000, 14000, 42000, 82000, 15000, 31000, 46000, 9000, 15000, 26000, 17000, 18000, 31000, 33000, 21000, 25000, 21000],
    "Dealer No": ["06457-3834", "60504-7114", "38701-8047", "99301-3882", "53546-9427", "85257-3102", "78758-7841", "78758-7841", "85257-3102", "85257-3102", "78758-7841", "78758-7841", "60504-7114", "38701-8047", "06457-3834", "60504-7114", "38701-8047", "99301-3882", "53546-9427", "85257-3102", "78758-7841", "85257-3102"],
    "Body Style": ["SUV", "SUV", "Passenger", "SUV", "Hatchback", "Hatchback", "Passenger", "Passenger", "Hardtop", "Passenger", "SUV", "Hardtop", "SUV", "Hatchback", "Sedan", "Hatchback", "Hatchback", "Passenger", "SUV", "Sedan", "Sedan", "Hardtop"],
    "Phone": [8264678, 6848189, 7298798, 6257557, 7081483, 7315216, 7727879, 6206512, 7194857, 7836892, 7995489, 7288103, 6842408, 7558767, 7677191, 8431908, 7814646, 7456650, 7627010, 6736704, 7889827, 6183219],
    "Dealer Region": ["Middletown", "Aurora", "Greenville", "Pasco", "Janesville", "Scottsdale", "Austin", "Austin", "Pasco", "Scottsdale", "Austin", "Pasco", "Aurora", "Greenville", "Middletown", "Aurora", "Greenville", "Pasco", "Janesville", "Scottsdale", "Austin", "Austin"]
}

df = pd.DataFrame(data)

# Selecting the numerical columns for clustering
X = df[["Annual Income", "Price ($)"]]

# Running K-means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Adding the cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income'], df['Price ($)'], c=df['Cluster'], cmap='viridis', edgecolor='k', s=100)
plt.xlabel('Annual Income')
plt.ylabel('Price ($)')
plt.title('K-means Clustering (K=4)')
plt.grid(True)
plt.show()
