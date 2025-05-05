import json

# Sample user data
users_data = {
    "user1": {
        "pin": 1234,
        "balance": 220000
    },
    "user2": {
        "pin": 5678,
        "balance": 150000
    },
    "user3": {
        "pin": 9101,
        "balance": 300000
    },
    "user4": {
        "pin": 1121,
        "balance": 50000
    },
    "user5": {
        "pin": 3141,
        "balance": 75000
    },
    "user6": {
        "pin": 5161,
        "balance": 120000
    }
}
# Write to users.json
with open('users.json', 'w') as file:
    json.dump(users_data, file, indent=4)
print("users.json created with sample user data.")
