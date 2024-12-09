import csv
import faker
import random

# Initialize Faker to generate random data
fake = faker.Faker()

# Function to generate a random job listing
def generate_job_listing(job_id):
    return [
        job_id,
        fake.company(),
        fake.job(),
        fake.city(),
        random.randint(50000, 120000),
        fake.text(),
        random.choice([0, 1])  # 0 for non-fraudulent, 1 for fraudulent
    ]

# CSV file header
header = ["Job_ID", "Company_Name", "Job_Title", "Location", "Salary", "Description", "Fraudulent"]

# Number of rows
num_rows = 300

# Generate data and write to CSV file
with open("sample_data.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i in range(1, num_rows + 1):
        writer.writerow(generate_job_listing(i))

print("CSV file generated: job.csv")
