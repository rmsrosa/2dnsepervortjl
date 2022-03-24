# Github actions

* If you want the github workflow to (re)build the whole site when deploying it, you should have `Build_and_Deploy.yml` under `.github/workflows`, so it runs when new commits are pushed to `main`. You should also have `Deploy.yml` stored away, say in `.github`, so it is not used. It is also recommended to have `__site/` included in the `.gitignore` file, so it is not duplicated in the repo.

* If, however, you don't want to rebuild the site and simply use the site that was built locally, then you should have `Deploy.yml` under `.github/workflows` so it runs when new commits are pushed to `main`. This action simply moves the contents of `__site/` in the main branch to the root of the `gh-pages` branch, to serve the website.

