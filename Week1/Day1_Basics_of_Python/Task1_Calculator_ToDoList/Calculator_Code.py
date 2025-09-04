
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y

def calc():
    while True:
        print("Simple Calculator")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "5":
            print("Exiting calculator...")
            break

        x = float(input("Enter first number: "))
        y = float(input("Enter second number: "))

        if choice == "1":
            print("Result:", add(x, y))
        elif choice == "2":
            print("Result:", subtract(x, y))
        elif choice == "3":
            print("Result:", multiply(x, y))
        elif choice == "4":
            if y != 0:
                print("Result:", divide(x, y))
            else:
                print("Error: Division by zero!")
        else:
            print("Invalid choice! Please enter 1-5.")

        print()

calc()
