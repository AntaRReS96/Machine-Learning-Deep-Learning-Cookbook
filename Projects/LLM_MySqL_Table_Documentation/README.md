# LLM MySQL Table Documentation

## Opis

Projekt do automatycznego generowania dokumentacji tabel MySQL z użyciem SQLAlchemy i pandas.

## Description

A project for automatically generating MySQL table documentation using SQLAlchemy and pandas.

## Wymagania

- Python 3.7+
- MySQL Server
- Pakiety Python wymienione w `requirements.txt`

## Requirements

- Python 3.7+
- MySQL Server
- Python packages listed in `requirements.txt`

## Instalacja

1. Sklonuj repozytorium lub pobierz pliki projektu
2. Zainstaluj wymagane pakiety: `pip install -r requirements.txt`

## Installation

1. Clone the repository or download project files
2. Install required packages: `pip install -r requirements.txt`

## Konfiguracja

Zaktualizuj zmienne połączenia MySQL:
- MYSQL_HOST - adres serwera
- MYSQL_PORT - port (domyślnie 3306)
- MYSQL_USER - nazwa użytkownika
- MYSQL_PASSWORD - hasło do bazy danych
- MYSQL_DATABASE - nazwa bazy danych

## Configuration

Update MySQL connection variables:
- MYSQL_HOST - server address
- MYSQL_PORT - port (default 3306)
- MYSQL_USER - username
- MYSQL_PASSWORD - database password
- MYSQL_DATABASE - database name

## Funkcje

- **Proste łączenie z MySQL**: Standardowe SQLAlchemy create_engine
- **Obsługa różnych schematów**: Przełączanie między bazami danych
- **Bezpieczne zarządzanie połączeniami**: Automatyczne zamykanie połączeń
- **Integracja z Pandas**: Bezpośrednie zwracanie DataFrame'ów

## Features

- **Simple MySQL connection**: Standard SQLAlchemy create_engine
- **Multiple schema support**: Switch between databases
- **Safe connection management**: Automatic connection closing
- **Pandas integration**: Direct DataFrame returns

## Bezpieczeństwo

⚠️ **Uwaga**: Nigdy nie umieszczaj haseł bezpośrednio w kodzie. Użyj zmiennych środowiskowych lub plików konfiguracyjnych.

## Security

⚠️ **Warning**: Never put passwords directly in code. Use environment variables or configuration files.

## Rozwiązywanie problemów

### Najczęstsze błędy:
- **"Access denied"**: Sprawdź dane logowania
- **"Can't connect to MySQL server"**: Sprawdź czy serwer MySQL jest uruchomiony
- **"Unknown database"**: Sprawdź nazwę bazy danych

## Troubleshooting

### Common errors:
- **"Access denied"**: Check login credentials
- **"Can't connect to MySQL server"**: Check if MySQL server is running
- **"Unknown database"**: Check database name

## Wsparcie

W przypadku problemów sprawdź:
1. Instalację wymaganych pakietów
2. Poprawność danych połączenia MySQL
3. Uprawnienia użytkownika do bazy danych

## Support

In case of problems check:
1. Required packages installation
2. MySQL connection data correctness
3. User database permissions 