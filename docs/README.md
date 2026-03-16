# Documentation workflow

From the `docs/` directory, regenerate the API reference files and rebuild the HTML docs with:

```bash
make apidoc SPHINXAPIDOC=../venv/bin/sphinx-apidoc
make html SPHINXBUILD=../venv/bin/sphinx-build SPHINXAPIDOC=../venv/bin/sphinx-apidoc
```

What these do:

- `make apidoc ...` regenerates the `.rst` API source files from `src/grf_gp`.
- `make html ...` reruns `apidoc` and rebuilds the HTML site in `docs/build/html`.

After changing Python modules, package structure, or docstrings, rerun `make html ...` to refresh the generated documentation.
