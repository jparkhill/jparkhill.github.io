---
name: connect-google-workspace
description: Walk a Google Workspace administrator through connecting their whole domain by domain-wide delegation (Google service reads and approved Chat sends for every member, with no sign-in by the members), one stage at a time, checking each stage with connector_setup_status. Use when an administrator asks to connect the firm's or the domain's mail, every member's mailbox, Workspace, a service account or domain-wide delegation, or when a delegated sync failed with unauthorized_client.
tools: [connector_setup_status]
requires: []
---
# Connect a Google Workspace domain (domain-wide delegation)

You guide; the administrator does every step. The Cloud Console and Admin
console steps happen in their browser, the key is stored from their terminal,
and no credential ever passes through this chat. The facts come from
`wiki/internal/connectors.md` (section *Domain-wide delegation*); if that page
and this skill disagree, the page wins. For one person connecting their own
account, use the `connect-google` skill instead.

What this sets up: a service account whose client id the administrator
authorises in the Admin console for eight read scopes and one Chat create scope. Durian then
asks Google for a short-lived token *acting as* a member (no sign-in by the
member), reads that member's Google data, and sends Chat only through the approved write tool.

## Rules

1. **Never ask for, and never accept, the service-account key in chat**: not
   the JSON file, not its `private_key`, not a screenshot of it. The key is
   read only by `durian connectors google-delegation` from a file, stdin or
   its hidden prompt. The client id and the service account's email are not
   secret; the administrator may say them.
2. **If key material is pasted anyway**, say at once: "That looks like a
   service-account key. I have not used it and will not. It is now in the chat
   history and was sent to the model provider, so treat it as exposed: delete
   that key and create a new one before going on." How: Cloud Console → *IAM &
   Admin → Service accounts* → the account → *Keys* → delete the key, then
   stage 4 again. Do not repeat it, store it or pass it to any tool.
3. **Check, do not assume.** Call `connector_setup_status(domain=<domain>)` at
   the start and again after every stage, and read its answer before moving
   on. When its "Next action" differs from what you were about to say, follow
   the tool.
4. **Say plainly what you cannot see.** The tool reads Durian's stored state
   only. It cannot see the Cloud project, whether the APIs are enabled, or the
   Admin console's delegation entry. It learns whether the authorisation is
   right only from a sync: Google refuses with `unauthorized_client` when the
   client id or a scope is missing. For stages 1 to 3 and 5 the tool's answer
   will not change; tell the administrator that, and that you rely on their
   word.
5. One stage per message. Wait for the administrator to say it is done.
6. You do not run `durian connectors google-delegation` yourself, with any
   tool. It needs the administrator's terminal and the key file.
7. Only a **Workspace super administrator** can do stage 5. If the person is
   not one, stop there and say who has to do it.

## Before the first stage

Ask for the Workspace domain (for example `example.test`; that is `<domain>`
below) and one member address to start with (`<member>`). Call
`connector_setup_status(domain=<domain>)` and start at the first stage that is
not done:

| Tool says | Start at |
|---|---|
| key not stored, no delegation record | stage 1 |
| key stored, record present, no member accounts | stage 6 with `--member` |
| `admin_authorised` not done and a refusal (`unauthorized_client`) shown | stage 5 |
| member accounts registered, a source never synced | stage 7 |
| every stage done | nothing to set up |

## Stages, in order

Stages 1 to 4 are in **Google Cloud Console**, <https://console.cloud.google.com>,
signed in as a Workspace administrator.

### 1. Cloud: project
Project picker → *New project*, any name (for example `durian-workspace`),
inside the Workspace organisation. Select it before going on.

### 2. Cloud: enable the required APIs
*APIs & Services → Library*: **Gmail API** → *Enable*; **Google Calendar API**
→ *Enable*; **Admin SDK API** → *Enable* (the member directory); also enable **Drive**,
**Docs**, **Sheets** and **Chat** APIs. Configure the Chat app for user authentication. A missed API
shows up later as a sync error `accessNotConfigured`.

### 3. Cloud: service account
*IAM & Admin → Service accounts → Create service account*. Name it (for
example `durian-reader`). Grant it **no** project roles and no user access:
delegation, not IAM, is what lets it read mail. On its details page note two
values, both non-secret: the **email** (`…@….iam.gserviceaccount.com`) and
the **Unique ID** / **OAuth 2 Client ID** (a long number). The client id is
needed in stage 5.

### 4. Cloud: create and download a JSON key
The service account → *Keys → Add key → Create new key → JSON*. The browser
downloads one file. It holds the private key: **do not send it here**, do not
put it in the repository, a shared drive or a chat. Keep it for stage 6.
(If the organisation policy `iam.disableServiceAccountKeyCreation` blocks
this, an organisation administrator must allow keys for this project first.)

Stage 5 is in the **Google Admin console**, <https://admin.google.com>, as a
super administrator.

### 5. Admin console: domain-wide delegation
*Security → Access and data control → API controls → Manage Domain Wide
Delegation → Add new*.
- **Client ID**: the service account's client id from stage 3 (the number,
  not the email).
- **OAuth scopes**: exactly this line, comma-separated, nothing more:

```
https://www.googleapis.com/auth/gmail.readonly,https://www.googleapis.com/auth/calendar.readonly,https://www.googleapis.com/auth/admin.directory.user.readonly,https://www.googleapis.com/auth/drive.readonly,https://www.googleapis.com/auth/documents.readonly,https://www.googleapis.com/auth/spreadsheets.readonly,https://www.googleapis.com/auth/chat.spaces.readonly,https://www.googleapis.com/auth/chat.messages.readonly,https://www.googleapis.com/auth/chat.messages.create
```

  Only chat.messages.create is write; all others are read-only. Durian never asks for any other scope and refuses
  to; a scope left out here makes that source fail with `unauthorized_client`
  and the tool names it. *Authorise*. Google can take a few minutes, rarely
  up to a day, before the authorisation takes effect.
- **I cannot see this page.** Ask the administrator to read back the client id
  and the scope list shown in the entry.

### 6. Terminal: store the key and register a member
Tell the administrator to run this **in a terminal on main**, not in chat:

```bash
durian connectors google-delegation --domain <domain> --key-file <path to the downloaded JSON> --member <member>
```

(Without `--key-file` it asks for the JSON at a hidden prompt; `--stdin`
reads it from a pipe. `--member` may be repeated.) It prints the secret's
name `google_delegation_key__<domain>` and its fingerprint, the service
account and client id it recorded, each member account registered, and the
scope line of stage 5. It refuses a file that is not a service-account key, a
member outside `<domain>`, and stores nothing then. **Then delete the
downloaded key file**: Durian keeps its own copy in the vault.

Call `connector_setup_status(domain=<domain>)`. Done means: key stored = yes
(with a fingerprint), delegation recorded with the client id from stage 3,
the member listed under member accounts.

### 7. Terminal: first sync
`durian connectors sync` in a terminal on main (or wait for the 15-minute
schedule). Then call `connector_setup_status(domain=<domain>)`.

| Tool shows | Meaning | What to do |
|---|---|---|
| every member account synced, `admin_authorised` done | Google issued read tokens; Drive metadata and Chat messages also sync; Docs/Sheets content is on demand; sending untested | finish |
| `unauthorized_client`, a scope named | stage 5 is missing that scope, or has the wrong client id, or has not taken effect yet | stage 5: compare client id and scope line; wait a few minutes; sync again |
| `invalid_grant` | the member does not exist in the domain, is suspended, or the key was deleted in Cloud Console | check the address; stage 4 if the key is gone |
| `outside_domain` | the address is not of `<domain>` | use a `<domain>` address |
| `key_missing` | the vault no longer holds the key | stage 6 again (stage 4 first if the file is gone) |
| `accessNotConfigured` | an API is not enabled | stage 2 |

### 8. Terminal: every member from the directory (optional)
To read every member instead of the ones named with `--member`, the
administrator gives the address of a Workspace administrator whose rights
list the directory (often their own), and keeps named mailboxes out:

```
durian connectors directory-exclude --domain <domain> --add <mailbox to keep out>
durian connectors directory-sync --domain <domain> --admin-email <a Workspace admin's address>
```

Each listed member gets Gmail, Calendar, Drive, Docs, Sheets and Chat accounts: owned by the Durian
user whose address it is (`durian user email <name> --email <address>`),
otherwise UNASSIGNED (the installation's admins manage it), and granted to
the person who connected the domain. The 15-minute schedule lists the
directory again: joiners are added, leavers disabled (their stored mail
stays). `durian connectors status` shows OWNER, GRANTED TO, ITEMS and LAST OK
per account. A 403 `notAuthorized` means the address given is not a
Workspace administrator.

### Finish
When the tool reports every stage done, say so with its numbers (members,
last sync and items per account). Say what stays outside the tool's view: the
Cloud project and the Admin console entry; revoking is deleting the entry in
stage 5 or the key in stage 4.

## Removing access

Deleting the delegation entry (stage 5) stops new tokens at once; tokens
already issued run out within an hour and are held only in Durian's memory.
Deleting the key in Cloud Console has the same effect. Disable the member
accounts in Durian so the schedule stops trying.
