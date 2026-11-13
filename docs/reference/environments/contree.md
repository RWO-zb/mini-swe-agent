# ConTree

!!! note "ConTree Environment class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/contree.py)
    - Requires [ConTree](https://contree.dev) token

    ??? note "Full source code"

        ```python
        --8<-- "src/minisweagent/environments/extra/contree.py"
        ```

::: minisweagent.environments.extra.contree

This environment executes commands in [ConTree](https://contree.dev) sandboxes using [ConTree SDK](https://github.com/nebius/contree-sdk)

## Setup

1. Install the dependencies:
   ```bash
   pip install "mini-swe-agent[contree]"
   ```

2. Set up ConTree token and base_url:
   ```bash
   export CONTREE_TOKEN="your-contree-token"
   export CONTREE_BASE_URL="your-given-base-url-for-contree"
   ```

{% include-markdown "../../_footer.md" %}

