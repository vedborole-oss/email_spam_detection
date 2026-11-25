# Publishing this project on GitHub (macOS)

Do these steps once. Total time: about 10 minutes.

## 0. Prerequisites
```bash
git --version          # if missing, macOS offers to install the developer tools; click Install
git config --global user.name  "YOUR_NAME"
git config --global user.email "the-email-you-used-for-github@example.com"
```
Create a free account at https://github.com if you do not have one.

## 1. Put your name and username in the files
Run from inside the project folder:
```bash
python3 scripts/personalize.py --name "YOUR_NAME" --username YOUR_GITHUB_USERNAME
```
This replaces the `YOUR_NAME` and `YOUR_GITHUB_USERNAME` placeholders in the README, LICENSE,
CITATION.cff, pyproject.toml and CONTRIBUTING.md.
(Replace the capitalised words in the command above with your real name and username first.)

## 2. Check that everything works before you publish
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[app,dev]"
ruff check .
pytest
spam-detector train
```
All checks must pass and `spam-detector train` must finish without errors.

## 3. Create the empty repository on GitHub
1. Open https://github.com/new
2. Repository name: `spam-detector`
3. Description: `SMS and email spam detection with TF-IDF, Naive Bayes, Logistic Regression and SVM`
4. Choose **Public**.
5. Do **not** tick "Add a README", ".gitignore" or "license" (the project already has them).
6. Click **Create repository**.

## 4. Upload your project
```bash
git init
git add .
git status                 # check: no venv/ folder and no models/*.joblib file should be listed
git commit -m "Initial commit: spam detection project"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/spam-detector.git
git push -u origin main
```
If GitHub asks for a password, use a **Personal Access Token** instead of your account password:
GitHub > Settings > Developer settings > Personal access tokens > Tokens (classic) > Generate new token,
tick `repo`, and paste the token when asked for the password. (Alternatively install GitHub CLI with
`brew install gh` and run `gh auth login`.)

## 5. After the first push
- Open the **Actions** tab. The CI workflow should start and turn green within a few minutes.
- On the repository page click the gear icon next to **About** and add:
  - Topics: `machine-learning`, `spam-detection`, `nlp`, `tf-idf`, `scikit-learn`, `python`, `streamlit`
- Create a release: **Releases > Create a new release**, tag `v1.0.0`, title `v1.0.0`, and paste the
  1.0.0 section of `CHANGELOG.md`.

## 6. Good habits for later changes
```bash
git checkout -b my-change        # work on a branch
# ... edit files ...
ruff check . && pytest
git add -A && git commit -m "Describe the change"
git push -u origin my-change     # then open a Pull Request on GitHub
```

## Note about the trained model
`models/*.joblib` is deliberately listed in `.gitignore`, because trained files should be recreated, not stored
in git. Anyone who clones the repository runs `spam-detector train` once (under a minute) and then everything
works. The dataset itself is included (about 0.5 MB) so the project runs offline.

## Putting it on your CV / resume
> Built an SMS/email spam classifier (TF-IDF with Naive Bayes, Logistic Regression and SVM) reaching 0.99 accuracy and
> 0.95 F1 on a held-out test set; added cross-validated tuning, a CLI, a Streamlit app, unit tests and GitHub Actions CI.
