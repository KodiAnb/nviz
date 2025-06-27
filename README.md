# nviz

## Deployment Notes:

In GitHub Pages, go to Settings -> Pages, and then Create Your Own Workflow.  Copy the contents of `main.yml` into this file and call it `main2.yml`.  Commit this.  Set the Build Action to `GitHub Actions` as well.  Add any dependencies to `package.json`.  Then, go to `Actions` and run the workflow, if it does not automatically trigger on checkin.  Delete `main2.yml` file when done.
