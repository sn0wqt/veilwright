# Project Developer Log: DataShield

## Overview
DataShield is a framework for detecting and anonymizing sensitive data in relational databases. The system separates the PII (Personally Identifiable Information) Scanner from the Anonymization Engine so each component can evolve independently. This document outlines the technical decision and refinements made during development.

## Member Responsibilities
**Asaad Falah Rasul** 
Led the core architectural design and backend implementation. Developed the Anonymization Engine, focusing on maintaining relational integrity through synchronized synthetic identity generation and batch-update systems. Implemented the two-tier PII Scanner using regular expressions and statistical heuristics.

**Ramadan Hatab**
Led Quality Assurance and stability improvements. Developed the test suite, identified critical database-locking bugs in the engine, and implemented reliable batch processing techniques to resolve local database deadlocks. Initially drafted the Developer Log.

**Dimitri Virginia Azzahra**
Responsible for project documentation and written reporting. Revised and reorganized the Developer Log, clarified descriptions of the architecture and implementation decisions, corrected inconsistencies in terminology, and checked that the report met the required submission criteria.

## Group Collaboration
At the beginning of the project, the group agreed to divide work based on prior experience. No major changes were introduced to this collaboration model during development. The division of tasks worked well and matched each member’s strengths. Although the group was originally formed with four members and continued with three, the workload was managed within the team. The group does not believe a different collaboration model would have significantly improved the process.

## Initial Project Specification
The initial project idea was to develop an automated anonymization tool for structured datasets containing sensitive information. The system would accept raw data files such as CSV, Excel, or JSON files containing personal or financial information including names,addresses, and IBAN numbers.

The agreed objective was to generate a sanitized version of the dataset where sensitive atributes are replaced with realistic synthetic values rather than removed. The replacement process should maintain structural integtity so that the resulting dataset can still be used for statistical analysis, visualization, and reporting.

The tool was expected to preserve database structure and the primary use case focused on producing data suitable for public reporting or company statistics without exposing private information.

-----

## Architectural Design and Dependency Management
The project was done with a focus on modern Python development standards. An early technical decision involved migrating `poetry` to `uv` for dependency management. The CLI infrastructure and SQLite connection logic were designed to support automated table initialization.

The system architecture consists of two primary modules:
1. **The Anonymization Engine**: Replaces sensitive fields with high-fidelity synthetic data and preserves relational consistency across tables.
2. **The PII Scanner**: Scans database schemas and column content to automatically identify potential PII using statistical heuristics and validation logics.

-----

## Implementation and Problem-Solving

### Core Data Models and Engine Stability
Development began with the defining structured database models, such as the `User` model, which incorporates multiple sensitive data fields. During scaling tests, the use of `yield_per` caused SQLite locking issues during concurrent write operations. The iteration logic was refactored to use OFFSET/LIMIT pagination to stabilize database access.

### Intelligent Scaling and Accuracy
The system evolved from basic metadata matching into a detection and masking framework. Key technical refinements included:
- **Identity Synchronization**: Implementation of the `generate_synchronized_identity` utility maintants that randomized usernames and emails remain correlated across complex relational schemas high-fidelity synthetic records.
- **Statistical Confidence**: The PII Scanner applies the Wilson lower bound for statistical confidence. This allows the engine to balance detection accuracy with row-level heuristics in unstructured datasets.
- **Support for Complex Schemas**: The framework was extended to handle composite primary keys and multi-database connection strings. Operation is no longer limited to SQLite and now includes PostgreSQL and MySQL.

To achieve mathematical confidence in these detections, we implemented the **Wilson score interval** to calculate a conservative lower bound for detection. A column receives "HIGH" confidence only when a significant and consistent sample of valid data is detected. Furthermore, we integrated the **Luhn algorithm** for credit card validation and specialized libraries for IBAN and phone number verification. These mechanism distinguish sensitive data from ambiguous headers with high accuracy.

### Anonymization Logic and Synchronized Identities
Synthetic data generation introduced uniqueness conflicts during large-scale username and email creation. Provider logic was updated to include bounded retry loops to prevent constraint violation.

This was eventually standardized by wrapping generators in an `ensure_unique` utility that verifies synthetic data against existing records before assignment.

To further preserve relational integrity, we implemented **Synchronized Identity** logic. When an identity is faked, the resulting `full_name`, `username`, and `email` are generated from a consistent template. 

We also refined context-aware logic; for instance, the **CVV generation** dynamically detects the card provider (e.g., American Express vs. Visa) to determine whether to produce a 3-digit or 4-digit code.

-----

## Optimization and Technical Refinement

* **Memory Management**: During bulk anonymization, we identified escalating memory usage that impacted large datasets. We resolved this by identifying and fixing memory leaks in the engine's core session handling logic .
* **Bulk Operations**: To improve performance on high-volume datasets, we implemented batch processing for database updates which reduces I/O overhead while maintaining transactional safety.
* **CLI Robustness**: Exception propagation was standardized. Critical failures (e.g., missing database files) are caught and reported via standard exit codes.
* **Financial Data Dynamics**: Generated financial fields follow structural rules and uniqueness constraints across datasets.
* **Quality Assurance**: Achieved full test coverage across the coverage across the core components. This required refining how our database connections and lifecycles to prevent resource leaks during teardown.

-----

## Use of Artificial Intelligence
Artificial intelligence tools were used in a limited capacity during development. AI tools was used to suggest small code completions and assist during code review.

-----