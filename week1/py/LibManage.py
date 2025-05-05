class User:
    def __init__(self, username):
        self.username = username
        self.borrowed_books = []
    def borrow_book(self, book_name):
        self.borrowed_books.append(book_name)
    def return_book(self, book_name):
        if book_name in self.borrowed_books:
            self.borrowed_books.remove(book_name)
            return True
        return False
class Library:
    def __init__(self):
        self.users = {}
        self.books = {
            "Book A": 3,
            "Book B": 2,
            "Book C": 5,
        }
    def register_user(self, username):
        self.users[username] = User(username)
    def login(self, username):
        return username in self.users
    def borrow_book(self, username, book_name):
        if book_name in self.books and self.books[book_name] > 0:
            self.users[username].borrow_book(book_name)
            self.books[book_name] -= 1
            return f"{username} has borrowed '{book_name}'."
        return "Book not available."
    def return_book(self, username, book_name):
        if self.users[username].return_book(book_name):
            self.books[book_name] += 1
            return f"{username} has returned '{book_name}'."
        return "You have not borrowed this book."
    def check_borrowed_books(self, username):
        return self.users[username].borrowed_books
# Main loop
library = Library()
while True:
    print("\n1. Register User")
    print("2. Login")
    print("3. Borrow Book")
    print("4. Return Book")
    print("5. Check Borrowed Books")
    print("6. Exit")
    
    choice = input("Choose an option: ")
    if choice == "1":
        username = input("Enter username: ")
        library.register_user(username)
        print("User registered successfully.")
    
    elif choice == "2":
        username = input("Enter username: ")
        if library.login(username):
            print("Login successful.")
        else:
            print("Invalid username.")
    
    elif choice == "3":
        username = input("Enter your username: ")
        book_name = input("Enter the name of the book to borrow: ")
        print(library.borrow_book(username, book_name))
    
    elif choice == "4":
        username = input("Enter your username: ")
        book_name = input("Enter the name of the book to return: ")
        print(library.return_book(username, book_name))
    
    elif choice == "5":
        username = input("Enter your username: ")
        borrowed_books = library.check_borrowed_books(username)
        print("Your borrowed books:", borrowed_books)
    
    elif choice == "6":
        print("Thank you for using the Library Management System. Goodbye!")
        break
    
    else:
        print("Invalid option. Please try again.")
