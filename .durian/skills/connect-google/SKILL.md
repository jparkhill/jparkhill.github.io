---
name: connect-google
description: Walk the operator through connecting a Google account (Google service reads and approved Chat sends), one stage at a time, checking each stage with connector_setup_status. Use when the operator asks to connect, set up, link or re-login Google, Gmail or Google Calendar, or when a "Google account needs login" notice arrived.
tools: [connector_setup_status]
requires: []
---
# Connect a Google account

You guide; the operator does every step. The Cloud Console steps happen in
their browser, the login happens in their terminal, and no credential ever
passes through this chat. The steps and their order come from
`wiki/internal/connectors.md` (sections 1 and 2); if that page and this skill
disagree, the page wins.

## Rules

1. **Never ask for, and never accept, a client secret, client id, authorization
   code, redirected address, access token or refresh token in chat.** Do not
   ask the operator to paste the downloaded JSON or a screenshot of it either.
   Those values are typed only at the hidden prompts of `durian connect google`.
2. **If one is pasted anyway**, say so at once, in these terms: "That looks
   like a credential. I have not used it and will not. It is now in the chat
   history and was sent to the model provider, so treat it as exposed: rotate
   it before going on." Then say how:
   - client secret: Cloud Console → *Credentials* → the client → *Reset secret*
     (or delete the client and create a new one), then
     `durian connect google --account <address> --reset-client`;
   - authorization code or redirected address: it is single-use and
     short-lived; start `durian connect google` again and do not reuse it;
   - refresh or access token: remove Durian's access at
     <https://myaccount.google.com/permissions>, then log in again.
   Do not repeat the value, do not store it with any tool, do not pass it to
   any tool. Continue only after the operator says it is rotated.
3. **Check, do not assume.** Call `connector_setup_status(account)` at the
   start and again after every stage, and read its answer before you move on.
   Go to the next stage only when the check (or, for a stage the tool cannot
   see, the operator's own word) says the current one is done. When the tool's
   "Next action" differs from what you were about to say, follow the tool.
4. **Say plainly what you cannot see.** The tool reads Durian's stored state
   only. It cannot see anything inside Google Cloud Console: not the project,
   not whether the APIs are enabled, and **not the consent screen's publishing
   status** (Testing or In production). It also cannot tell whether Google
   would accept the stored token right now. For stages 1 to 5 the tool's
   answer will not change; tell the operator that, and that you are relying on
   what they report.
5. One stage per message. Wait for the operator to say it is done.
6. You do not run `durian connect google` yourself, with any tool. It needs
   the operator's terminal and hidden prompts.

## Before the first stage

Ask which Google address is being connected; that address is `<address>` below
and the `account` argument of the tool. Call
`connector_setup_status(<address>)` and start at the first stage that is not
done:

| Tool says | Start at |
|---|---|
| OAuth client not stored | stage 1 |
| client stored, sign-in `needs_login` | stage 6 (after the 7-day question below if a login worked before) |
| sign-in valid, accounts not registered | stage 7 |
| accounts registered, a source never synced | stage 8 |
| every stage done | nothing to set up; remind them of the publishing status (stage 3) |

If the tool lists a different name under "Accounts with a stored refresh
token", ask whether that is the same account under another name before
starting over.

## Stages, in order

Stages 1 to 5 are in the browser at <https://console.cloud.google.com>.
Give the operator the direct link for each stage (below) with the exact words
on the button to press; do not just name the console.

### 1. Project
<https://console.cloud.google.com/projectcreate>.
Project picker → *New project*, any name (for example `durian-connectors`).
Select it before going on: everything below is per project.

### 2. Enable the six APIs
With the project selected, open each link and press *Enable* (one message per
link is fine; a missed one makes the later sync fail with 403
`accessNotConfigured` for that source):

- Gmail: <https://console.cloud.google.com/apis/library/gmail.googleapis.com>
- Calendar: <https://console.cloud.google.com/apis/library/calendar-json.googleapis.com>
- Drive: <https://console.cloud.google.com/apis/library/drive.googleapis.com>
- Docs: <https://console.cloud.google.com/apis/library/docs.googleapis.com>
- Sheets: <https://console.cloud.google.com/apis/library/sheets.googleapis.com>
- Chat: <https://console.cloud.google.com/apis/library/chat.googleapis.com>

**Chat needs one more step, or every Chat call returns 403 PERMISSION_DENIED:**
after enabling it, open the Chat API's *Configuration* tab
(<https://console.cloud.google.com/apis/api/chat.googleapis.com/hangouts-chat>)
and save a Chat app. The operator has no URLs of their own to give, and needs
none:

- App name: anything, for example `Durian`.
- Avatar URL: `https://developers.google.com/chat/images/quickstart-app-avatar.png`
  (Google's own sample avatar; any public https image works).
- Description: anything.
- **Turn *Enable interactive features* off.** With it off the page asks for no
  endpoint, HTTP or app URL. Durian only reads spaces and messages and sends
  approved messages as the signed-in user; it is not a bot that receives events.
- *Save*.

Chat is optional: if the operator would rather skip it, the other five sources
work without it and Chat simply stays "never synced".

### 3. OAuth consent screen
*APIs & Services → OAuth consent screen* (newer consoles: *Google Auth
Platform* with *Branding*, *Audience*, *Data access*).
- User type **External** for a gmail.com address (*Internal* exists only in a
  Google Workspace organisation).
- App name and support email: anything.
- Scopes (*Data access*): exactly
  `https://www.googleapis.com/auth/gmail.readonly` and
  `https://www.googleapis.com/auth/calendar.readonly`, plus the same URI prefix
  with `drive.readonly`, `documents.readonly`, `spreadsheets.readonly`,
  `chat.spaces.readonly`, `chat.messages.readonly` and `chat.messages.create`.
  Only the last permits writes. Durian refuses a grant
  that is broader or narrower and stores nothing.
- Test users (*Audience*): add `<address>`.
- **Publishing status: set it to In production now** (*Audience → Publish
  app*), before logging in. While an External app is in *Testing*, Google
  expires the sign-in 7 days after each login and Durian goes to
  `needs_login` every week. No verification is needed for personal use; the
  consent page will show "Google hasn't verified this app" (*Advanced → Go to
  … (unsafe)*). An *Internal* Workspace app has no such expiry.
  **I cannot see this setting.** Ask the operator to read the status off the
  page and tell you what it says.

### 4. OAuth client of type Desktop
*APIs & Services → Credentials → Create credentials → OAuth client ID* (or
*Clients → Create client*). Direct link: <https://console.cloud.google.com/apis/credentials>.
Application type **Desktop app**, not *Web
application*: only a Desktop client accepts the loopback redirect Durian uses.
A *Web application* client makes Google show "Access blocked: This app's
request is invalid" (Error 400 `redirect_uri_mismatch`) at stage 6; the fix is
a new Desktop client and `durian connect google --account <address> --reset-client`.

### 5. Download JSON
On the client's row, *Download JSON*. It holds the client id and client secret.
Tell the operator now what stage 6 will ask for, because the command never asks
for the file itself: two values from inside it, `client_id` (ends in
`.apps.googleusercontent.com`) and `client_secret` (usually starts `GOCSPX-`).
In the file they sit under the key `installed`; if the key is `web`, the client
is the wrong type (stage 4).
The operator keeps it open for the next stage and **does not send it here**.
After stage 6 succeeds they delete the file (not in `~/Downloads`, not in the
repository, not in a chat).

### 6. Log in, in a terminal
Tell the operator to run this **in a terminal on main (main-box)**, not in chat:

```bash
durian connect google --account <address>
```

What they will see: two hidden prompts for the client id and client secret
(skipped when a client is already stored), then instructions to approve
read access and Chat creation in a browser. With a browser on the same machine nothing is
pasted. With a browser on another machine they open the printed address there,
approve, and paste the address the browser was sent to **at the terminal's
hidden prompt** (the page failing to load is expected). It ends with
`Connected Google account …` and the names of three stored secrets, or with
`Not connected: <code>. Nothing was stored.`

They report only "connected" or the error code. Then call
`connector_setup_status(<address>)`. Done means: OAuth client stored = yes,
sign-in = valid, the required scopes verified after sync (older token metadata may be unknown). If not:

| Code | Meaning | What to do |
|---|---|---|
| `scope_mismatch` | a box was unticked on the consent page, or extra scopes were granted | run the command again and leave all required boxes ticked |
| `access_denied` | approval was refused on the consent page | again, and approve |
| `invalid_client` | the client id or secret typed does not match the client | again with `--reset-client` |
| `login_timeout` | no approval within 5 minutes; also what happens when Google shows an error page instead of redirecting (for example `<address>` is not a test user of an app in Testing) | fix what the Google page said (stage 3), then again |
| `port_in_use` | the loopback port is taken and there was no terminal to paste at | again with `--port 0` |

### 7. Accounts registered
The tool must show gmail, calendar, drive, docs, sheets and chat registered.
`durian connect google` registers all six itself after a successful login and
starts their first sync. If the tool shows any as not registered (a login
made before 2026-11-16, or the registration step failed), have the operator
run `durian connectors add google --account <name>` in a terminal, then check
again.

### 8. First sync
Tell the operator to run `durian connectors sync` in a terminal (or wait for
the 15-minute schedule). Call the tool. Done means each source has a last sync
time; report the item counts it gives. If a source shows a last error, read it
out and follow the tool's next action (`accessNotConfigured` → stage 2).

### Finish
When the tool reports every stage done, say so with the numbers (last sync and
items per source), and repeat the one thing you could not check: if the
consent screen is still in *Testing*, this sign-in ends about 7 days after the
login.

## Re-login (`needs_login` after it worked)

Ask when the last login was. About 7 days ago → almost certainly the Testing
status: stage 3 (publish), then stage 6. Otherwise the operator removed access,
changed their password, or deleted the client: stage 6 (stage 4 first if the
client is gone). Confirm with the tool as in stage 6.
