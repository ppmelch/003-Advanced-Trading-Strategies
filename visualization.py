from libraries import *


def plot_test_validation(port_value_test, port_value_val):
    """
    Plotea test y validation de forma continua:
    - Test empieza en 1,000,000
    - Validation arranca donde terminó test
    """
    plt.figure(figsize=(12, 6))

    base_value = 1_000_000

    # Normaliza test para iniciar en 1M
    test_scaled = port_value_test / port_value_test.iloc[0] * base_value

    # Escala val para que arranque donde terminó test
    val_scaled = port_value_val / port_value_val.iloc[0] * test_scaled.iloc[-1]

    # Ejes X continuos
    x_test = range(len(test_scaled))
    x_val = range(len(test_scaled), len(test_scaled) + len(val_scaled))

    # Plot
    plt.plot(x_test, test_scaled, label="Test", color="royalblue", lw=2)
    plt.plot(x_val, val_scaled, label="Validation", color="orange", lw=2)

    plt.title("Test + Validation Portfolio", fontsize=14, fontweight="bold")
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

