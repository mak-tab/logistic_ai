git checkout --orphan clean-main
git add -A
git commit -m "initial"
git branch -D main
git branch -m main
git push -f origin main