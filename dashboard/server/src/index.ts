// Start the review API. Run from dashboard/server:
//   npm run dev          (REVIEW_DB and REPO_ROOT default to this repo's layout)

import path from "node:path";
import { fileURLToPath } from "node:url";
import { createApp } from "./app.js";
import { openDb } from "./db.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = process.env.REPO_ROOT ?? path.resolve(here, "../../..");
const dbPath = process.env.REVIEW_DB ?? path.join(repoRoot, "outputs/phase5/review.db");
const port = Number(process.env.PORT ?? 4317);

const app = createApp({ db: openDb(dbPath), repoRoot });
// Express 5 passes listen errors (e.g. port in use) to the callback instead of throwing.
app.listen(port, "127.0.0.1", (err?: Error) => {
  if (err) {
    console.error(`could not start on port ${port}: ${err.message}`);
    process.exit(1);
  }
  console.log(`review API on http://127.0.0.1:${port} (db ${dbPath})`);
});
