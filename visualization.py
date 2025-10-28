from libraries import *

def plot_tran_test_validation(port_value_train, port_value_test, port_value_val):
    plt.figure(figsize=(12, 6))

    portfolios = {
        "train": port_value_train,
        "test": port_value_test,
        "val": port_value_val
    }

    for name, values in portfolios.items():
        plt.plot(values, color=colors, marker="o", markersize=3, label=name)

    plt.title("Train, Test and Validation Portfolio", fontsize=14, fontweight="bold")
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_test_validation(port_value_test, port_value_val) -> None:
    """Plotea el portafolio de test + validación."""
    plt.figure(figsize=(12, 6))

    x_test = range(len(port_value_test))
    x_val = range(len(port_value_test), len(port_value_test) + len(port_value_val))

    plt.plot(x_test, port_value_test, label="Test", color="royalblue", lw=2)
    plt.plot(x_val, port_value_val, label="Validation", color="darkorange", lw=2)

    plt.title("Test + Validation Portfolio", fontsize=14, fontweight="bold")
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

