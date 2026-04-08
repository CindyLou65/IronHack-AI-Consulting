# LAB M.2.02 — Python for Low Code
## Refactoring Report | Path 1: Refactor Your Own Code

> **Source:** `product_generator_refactored.ipynb` | 43 cells | 24/24 audit checks passed

---

## 1. What Was Refactored

The starting point was the Jupyter notebook from Lab M.1.05 (OpenAI API calling), which generated product descriptions using GPT-4o Vision. The refactoring followed **Path 1** — improving code written by the student.

**The original code had these problems:**

- Monolithic functions doing everything at once (load + validate + call API + parse + save)
- Repeated logic: image encoding and JSON parsing duplicated across multiple cells
- Mixed concerns: API calls, parsing, and formatting all inside the same function
- Silent failures: errors caught with bare `except` but never shown
- Hardcoded logic and values that should have been parameters
- Helpers scattered across cells instead of centralised
- Dataset loaded twice — Cell 3 was unnecessary and was removed
- Global OpenAI client variable instead of passing it as a parameter

**The goal was code that is:** Maintainable, Testable, Modular, Reusable, and — above all — **Fails loudly, NOT silently.**

---

## 2. Helper Functions Created (Step 3)

All helpers were extracted into **ONE centralised cell (Cell 14)** so they can later be moved to `helpers.py` without changes. The cell includes its own imports, custom exceptions, and all six functions.

### Custom Exception Hierarchy

- `PipelineError(Exception)` — base class for all pipeline errors
- `ValidationError(PipelineError)` — raised when product data is invalid
- `ResponseParseError(PipelineError)` — raised when the model returns unparseable JSON

**Why:** enables structured catching (catch `ValidationError` separately from `APIError`), supports the "fail loudly" requirement, and makes the source of every failure immediately clear.

### Six Core Helper Functions

| Function | Single Responsibility | Tested With |
|---|---|---|
| `load_json_file(file_path)` | Load and parse JSON file safely | Expected failure on missing file |
| `validate_product_data(product_dict)` | Validate product fields via Pydantic `ProductSchema` | Pass + fail cases |
| `create_product_prompt(product, ...)` | Build Vision prompt with persona/style | Output preview |
| `parse_api_response(response)` | Strip markdown fences, parse JSON to dict | Valid + invalid JSON |
| `format_output(product, description)` | Assemble final output row dict | Structure check |
| `pil_to_base64_jpeg(pil_img)` | Encode PIL Image to base64 JPEG string | Encoding check |

> **Key insight from testing:** a helper is correct if it either returns the right output OR raises a clear, informative error. Silent success AND silent failure are both wrong.

---

## 3. How the Code Was Modularised (Step 4)

The original monolithic `process_products` function was broken into **eight single-responsibility functions**. Each function does exactly one job in the pipeline:

| # | Function | Single Job |
|---|---|---|
| 1 | `load_and_validate_products(products_df, n)` | Load from DataFrame + validate each product |
| 2 | `build_image_url_from_product(product)` | Encode image + create base64 data URL |
| 3 | `call_openai_for_listing(prompt, image_url, api_client)` | Call OpenAI API only — returns raw response |
| 4 | `generate_listing_once(product, api_client)` | One attempt: validate → prompt → encode → call → parse |
| 5 | `generate_listing_with_retries(product, api_client, max_retries)` | Retry logic only — wraps `generate_listing_once` |
| 6 | `process_one_product(product, api_client)` | Convert result into final structured output row |
| 7 | `run_batch(products_df, api_client, n)` | Orchestrator: loop → `process_one_product` → collect results |
| 8 | `save_results(results_df, jsonl_path, csv_path)` | Write JSONL (preserves lists) + CSV (flattens lists) |

**Critical improvement — API client passed as parameter, not a global:**

```python
# BEFORE (bad)
client = OpenAI(...)   # global variable, hidden dependency

# AFTER (good)
def call_openai_for_listing(prompt, image_url, api_client):
    ...
```

Passing `api_client` as a parameter improves testability (easy to mock), modularity (no hidden globals), and flexibility (swap client without changing function signatures).

---

## 4. Error Handling — Before and After

All error messages now follow the required 4-part format:

```
[function_name] ErrorType in location: message. Tip: suggestion.
```

| Error Type | How Handled | Cell |
|---|---|---|
| `FileNotFoundError` | Caught explicitly in `load_json_file`; shows full resolved path + fix tip | 14 |
| `JSONDecodeError` | Caught separately in `load_json_file` AND `parse_api_response`; exposes `e.lineno` and `e.colno` | 14 |
| `Pydantic ValidationError` | `ProductSchema.model_validate()` catches `PydanticValidationError`; re-raised as custom `ValidationError` with product name and field details | 14 |
| `openai.RateLimitError` / `APIConnectionError` | Caught by name in retry loop; triggers backoff + `logger.warning` (retryable) | 35, 37 |
| `openai.APIError` | Caught by name; fails fast with HTTP status code + tip (non-retryable) | 35, 37 |
| `ConnectionError` / `TimeoutError` | Caught alongside `APIConnectionError`; triggers retry with backoff | 35, 37 |

**Before — silent, useless:**
```python
except Exception as e:
    print(str(e))
# output: "Expecting value: line 1 column 1"
```

**After — loud, traceable:**
```
[parse_api_response] JSONDecodeError at line 1, col 1 in model output.
Raw preview: "```json\n{title: missing quotes}".
Tip: ensure prompt ends with "Return ONLY valid JSON".
```

---

## 5. Cleanup Done

The following were removed during refactoring:

- Duplicate image encoding helper cell (`pil_to_base64_jpeg` defined twice)
- Duplicate parsing helper `parse_api_response_text` (replaced by unified `parse_api_response`)
- Cell 3 — unnecessary second dataset load
- Redundant imports scattered across cells (consolidated into Cell 14)
- Orphan code block in Cell 34 — a detached `if`-block outside any function that would cause a `SyntaxError` on execution; replaced with an explanatory note

---

## 6. Extras Implemented

- **Retry logic:** `max_retries=2` loop in both `generate_listing_with_retries` (Cell 35) and `process_products` (Cell 37), with linear backoff on `RateLimitError`, `APIConnectionError`, `ConnectionError`, `TimeoutError`
- **Python logging module:** `import logging` with `logger.info` / `logger.warning` / `logger.error` replacing all `print()` statements in pipeline cells — log level and format configurable
- **Custom exception hierarchy:** `PipelineError` → `ValidationError` / `ResponseParseError` — enables targeted catching and cleaner test assertions
- **Pydantic `ProductSchema`:** formal data contract with `id`, `productDisplayName`, `gender`, `articleType`, `image` (`Any`) — uses `model_config = ConfigDict(arbitrary_types_allowed=True)` for PIL compatibility

---

## 7. Challenges Faced

- **Shell escaping complexity:** writing multi-line Python inside `-c` shell commands caused repeated quoting failures on Windows; solved by switching to a temporary `_patch_notebook.py` script
- **Windows encoding:** `sys.stdout` defaulted to `cp1252`, causing `UnicodeEncodeError` on emoji characters in print output; fixed with `sys.stdout.reconfigure(encoding='utf-8')`
- **Cell index drift:** inserting new cells shifted all subsequent indices; the patch script recalculated indices after each insert to avoid overwriting the wrong cells
- **Pydantic + PIL:** `PIL Image` is not a standard Pydantic-compatible type; required `model_config = ConfigDict(arbitrary_types_allowed=True)` inside `ProductSchema`
- **Orphan code detection:** Cell 34 had a detached `if`-block at module level (leftover from an earlier edit) that would crash the kernel silently — only caught during the audit

---

## 8. What Was Learned

- **Single Responsibility Principle:** each function does exactly one job — independently testable and easier to debug without running the full pipeline
- **Fail loudly, not silently:** a clear error message with function name, error type, location, and a tip is far more valuable than a caught exception that disappears
- **Separate concerns:** load / validate / generate / parse / format / save each belong in their own function — mixing them creates hidden coupling that breaks when one part changes
- **Pass dependencies as parameters:** injecting the API client rather than using a global removes hidden coupling and makes mocking easy in tests
- **Explicit error types matter:** `RateLimitError` should retry; `AuthenticationError` should fail fast — catching everything as `Exception` loses this distinction
- **Centralise helpers:** defining all helpers once in one cell (later one file) prevents drift where two cells have slightly different versions of the same logic

---

*LAB M.2.02 | product_generator_refactored.ipynb | 43 cells | 24/24 audit checks passed | Path 1: Refactor Your Own Code*
