# DataShield

DataShield is a consistent data anonymization tool designed to help users manage and protect sensitive information. This project provides a set of utilities and models for data anonymization, ensuring that data privacy is maintained while allowing for data analysis and processing.

## Installation

```bash
# Install uv if you haven't already
pip install uv

# Sync dependencies
uv sync --extra dev

```

## Usage

DataShield operates via a command-line interface. You can append `--help` to any command to see all available options, arguments, and default values.

```bash
# View all available commands
datashield --help

# View specific documentation for a single command
datashield anonymize --help

```

### Common Commands

Initialize the SQLite database tables:

```bash
datashield init-db --db-path datashield.db

```

Run a health check on the database connection:

```bash
datashield check --db-path datashield.db

```

Seed the database with a specific number of test users:

```bash
datashield seed --count 50 --db-path datashield.db

```

Scan the database for PII columns and generate a report:

```bash
datashield scan --db-path datashield.db --show-all

```

Anonymize the detected PII in the database:

```bash
datashield anonymize --db-path datashield.db --output anonymized.db --aggressive

```

Run a dry-run to see an anonymization plan without making any actual changes:

```bash
datashield anonymize --db-path datashield.db --dry-run

```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
