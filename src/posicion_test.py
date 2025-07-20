# test_posiciones.py
from ig_api import IGClient

def main():
    cliente = IGClient()
    cliente.login()
    cliente.get_open_positions()

if __name__ == "__main__":
    main()
