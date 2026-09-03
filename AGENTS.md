# Instructions for Agents (opencode)

- **Compilation Verification:** Always, without exception, you must try to compile the code (`cargo check` or `cargo build` as appropriate) and ensure there are no errors before finalizing changes in code files.
- **Daemon Deployment:** It is not necessary to restart the daemon or copy the executable file (`sudo systemctl restart...` or `sudo cp...`), the user manually handles that step.

- **Patterns:**
I am a fan of Rust abstractions and enjoy enums when they make sense. Try to use patterns that improve maintainability and minimize the state space of the program.
