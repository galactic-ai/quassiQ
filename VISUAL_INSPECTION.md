# Running the visual inspection viewer on Lonestar6

One person runs the viewer on Lonestar6. Each reviewer connects to that same server through SSH and uses their own name in the viewer. You can reopen the browser tab while the server and your SSH tunnel are running.

## Start the viewer

1. Put `index.html` and `build_vi_dataset.py` in the same directory on Lonestar6. Use the same directory each time so the viewer can find the existing score database.
2. From a terminal on your own computer, log in with port forwarding. Replace `YOUR_TACC_USERNAME` with your username.

   ```bash
   ssh -L 8000:127.0.0.1:8000 YOUR_TACC_USERNAME@ls6.tacc.utexas.edu
   ```

3. In that SSH session, go to the directory containing the viewer files. Run `hostname -f` and copy the specific login-node hostname to share with reviewers. Then run the viewer command. Replace the redshift catalog path with the path to your copy of `post_chi2_calc.csv`.

   ```bash
   python3 build_vi_dataset.py \
     --plots /work/10579/prisha/ls6/desi_project/plots \
     --redshift-csv /path/to/post_chi2_calc.csv \
     --latent-csv /work/10579/prisha/ls6/desi_project/latent_info.csv \
     --subset-csv /work/10579/prisha/ls6/desi_project/final_subset.csv \
     --serve
   ```

4. Keep that terminal open. On your computer, visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/). Send reviewers the specific hostname and the joining instructions below.

## Join as a reviewer

1. Get the specific Lonestar6 login-node hostname from the person running the viewer. The generic Lonestar6 hostname can route you to a different node. [Lonestar6 SSH access](https://docs.tacc.utexas.edu/hpc/lonestar6/#secure-shell-ssh)
2. In Terminal on your own computer, use your TACC username and the host’s exact node name. The following command uses `login1` as an example.

   ```bash
   ssh -N -L 8000:127.0.0.1:8000 YOUR_TACC_USERNAME@login1.ls6.tacc.utexas.edu
   ```

3. Leave the terminal open after login. With `-N`, the terminal can look idle while the tunnel is running. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.
4. Enter your own reviewer name and click **Use name**. Use the same name each time to reopen your saved reviews. Click a score from **1–5** to save it along with the current note, or use **Save note** to save the note on its own. **Show other reviewers’ scores** also displays their notes for the current target.
5. To filter by latent spread, enter a cutoff beside each latent you want to use. Leave unused cutoffs blank and choose **Any entered cutoff** or **All entered cutoffs**.

## Resume after a disconnect

| What happened | What to do |
| --- | --- |
| You closed only the browser tab | Reopen [http://127.0.0.1:8000/](http://127.0.0.1:8000/). |
| Your SSH tunnel disconnected | Run your tunnel command again. |
| The Python viewer stopped | One person restarts it from the same directory. If it starts on a different Lonestar6 node, share the new hostname and reconnect the reviewer tunnels. |

The server saves shared scores and notes in `vi_scores.sqlite3` beside `dataset.json` by default. Keep that SQLite file and reuse it when restarting the viewer. If you launch the viewer from another directory, pass `--score-db /path/to/vi_scores.sqlite3` to reuse your existing reviews.
