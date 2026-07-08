### Grocery List Calculator

TAX_RATE = 0.065  # 6.5% tax


def get_item_costs():
    """Collects item costs from the user until they type 'done'."""
    costs = []

    while True:
        user_input = input("Enter item cost or type 'done': $")

        if user_input.lower() == "done":
            break

        try:
            cost = float(user_input)
            if cost < 0:
                print("Please enter a positive number.")
                continue
            costs.append(cost)
        except ValueError:
            print("Invalid input. Please enter a number or 'done'.")

    return costs


def calculate_total(costs, tax_rate):
    """Calculates total price including tax."""
    subtotal = sum(costs)
    total = subtotal * (1 + tax_rate)
    return subtotal, total


def run_calculator():
    """Run one full calculator session."""
    costs = get_item_costs()

    if not costs:
        print("No items entered.")
        return

    subtotal, total = calculate_total(costs, TAX_RATE)

    print(f"\nSubtotal: ${subtotal:.2f}")
    print(f"Total after tax: ${total:.2f}\n")


def main():
    """Main program loop allowing restart."""
    while True:
        run_calculator()

        again = input("Would you like to use the calculator again? (y/n): ").lower()
        if again != "y":
            print("End")
            break


if __name__ == "__main__":
    main()
