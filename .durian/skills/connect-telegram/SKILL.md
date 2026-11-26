---
name: connect-telegram
description: Guide a person from BotFather to a linked, answering Telegram chat, one stage at a time, checking stored state with telegram_setup_status. Use when someone asks to connect or resume Telegram setup.
tools: [telegram_setup_status]
requires: []
---
# Connect Telegram

You guide; the person does every real step in her own browser and terminal.
The administrator steps are hers, using her administrator access to this
installation (Workspace admin alone does not grant Durian administration).
If she lacks that access, this installation's administrator must issue the
invite and activate the service. Never run credential input or real Telegram
stages yourself. See `wiki/internal/users.md`, Guided Telegram setup.

## Rules and resume

- Never ask for or accept a bot token, join code, API token or screenshot of
  credentials in this chat. The bot token is never pasted in chat, including
  the new bot's chat: only at hidden terminal prompts. The one-time join code
  goes only to the bot's deterministic `/join` command, not to this agent.
- If a token appears here, do not repeat, store or use it. Say exactly:
  "That looks like a credential. I will not use it. It is now in chat history;
  revoke it with BotFather's /revoke, obtain a replacement and enter that only
  in your terminal." Never call a tool with the value.
- Work **one stage per message**. Wait for her completion, then call
  `telegram_setup_status(user="<name>")` before proceeding. When the tool's
  next action disagrees with what you were about to say, follow the tool.
- Use the person's existing Durian name (confirm the existing name in the
  user listing if needed). Do not create a duplicate user. At the start call
  the tool. Its `stage` is the first unfinished stage. Token already stored:
  skip BotFather creation; if validated, use `bot_username` to name the bot.
  This installation has one bot shared by its linked users, not one per user.
- The tool reads names/metadata only and never calls Telegram. BotFather
  creation is invisible: after she says she has the token, move from stage 1
  to stage 2 even though stored state has not changed. A token stored but not
  validated always resumes at stage 2. No token or code is ever in the output.
- A linked chat is not proof she saw a reply. Stage 5 requires her own
  confirmation; do not declare her personally connected from test results.

## 1. Create or locate the bot

Check: `durian telegram-status --user <name>` (or the status tool).
If `token_stored=true`, skip creation. Otherwise say:

"Open the verified @BotFather in Telegram, send /newbot, and follow its prompts
for a display name and a unique username ending in bot. Keep the token private.
Do not paste it here. Tell me only when you have it ready in your own terminal."

If BotFather rejects a username, choose another ending in `bot`. If she already
created this installation's bot, use that bot's token; do not create another.
Check remains unchanged until stage 2; her confirmation is the stage 1 check.

## 2. Store and validate, hidden terminal input

Say: "In your own terminal connected to this installation, run
`durian connect telegram`. Enter the bot token at each of the two hidden
prompts. Tell me only 'validated' or the fixed error code; never the token."

Advanced deterministic provisioning: `durian connect telegram --stdin` reads
exactly two lines, token then the same token for confirmation, from a private
pipe. Never put either in argv, a shell command, shell history, an environment
variable or a repository file. Prefer the hidden prompts for a person.

The command writes service-tier `telegram_bot_token` through SecretStore and
makes one getMe request. It reports `validated` and the public username, never
the credential. The CLI must target the installation that runs this bot:
`durian status` checks the logged-in service. Without a remote login, run the
command on the main host while the service is stopped (local DB access).

Check: `durian telegram-status --user <name>`. Done means
`token_stored=true`, `bot_validated=true` and `bot_username` is present.
`malformed_token`: copy the whole token again at the hidden prompts.
`tokens_do_not_match`: re-enter both prompts. `hidden_terminal_required`: use
her own interactive terminal. `get_me_failed`: the token was stored but was
not validated; check connectivity, then obtain the current token from
BotFather (revoke/replace if exposed) and rerun. No raw Telegram error is shown.
A getMe success is historical proof only: later revocation is invisible here.

## 3. Activate the channel

Say: "The token is validated. Your administrator step is to start or restart
this installation so it loads the token. Then we'll check that it is polling."

For a running service, the administrator uses the recorded restart endpoint
from her terminal; this code reads her saved client config without displaying
its credential. Do not run this yourself inside an agtask:

```bash
uv run python - <<'PY'
import asyncio
from durian.remote import RemoteClient
async def main():
    result = await RemoteClient.from_config().request(
        "POST", "/service/restart", {"when_idle": True})
    print(result)
asyncio.run(main())
PY
```

For an installation not yet running, in its main checkout and the
administrator's terminal: `durian serve` (or its installed service manager's
start procedure). Do not start a second copy of a running service.
No hand-edited config file is needed: startup reads the vault token and enables
Telegram. A legacy `channels.telegram.botToken` overrides the vault; an admin
must remove that legacy plaintext value if present, without displaying it.

Check: `durian status`, then `durian telegram-status --user <name>`.
`polling_recently=true` means the running channel observed a successful getUpdates in memory in the last
120 seconds for the validated token. Wait for one long poll after activation.
If false, check the installed service health; pending idle restart means wait.
If still false, resolve service startup/network/competing-poller errors or
return to stage 2 if the token was revoked. Do not guess it is running.

## 4. Admit and join the existing user

Say: "Your chat is admitted by a successful /join; no separate allow-list edit
is needed. As administrator, run `durian user invite <name>` in your terminal.
Keep its one-time code out of this chat. Then open a private chat with
@<bot_username> and send /join followed by that code there."

If `join_code_pending=true`, skip issuing a new invite: use the existing code
before `join_code_expires_at`. If it was lost, the administrator issues another.
An already linked administrator may use `/invite <name>` in her private bot
chat instead. Only if `user_exists=false`, the administrator first runs
`durian user add <name>`; do not add a person who already exists.

Check: `durian telegram-status --user <name>` and `durian users`.
Done means her chat appears under `linked_chats`; the code is no longer pending.
Expired/wrong/used code: issue a new invite and retry in a private chat with the
named bot. "Not authorized": `/join` is permitted even before admission; send
it to that bot, without needing a config allow-list entry. Five wrong attempts
lock that chat's joins for an hour; wait rather than hammering it. Never remove
another person's linked chat or replace an allow-list with only her ID.

## 5. Confirm a real reply

Say: "Send /help to @<bot_username> from your linked private chat. Did the bot
answer with its commands? Tell me only whether you saw the reply."

Check: her actual reply confirmation, plus
`durian telegram-status --user <name>` showing her linked chat.
An administrator can also send `/users` and see her linked chat count;
ordinary users use `/help` because `/users` is administrator-only.
If there is no reply, check stage 3 and that she opened the recorded bot.
If her chat is not linked, return to stage 4. If both checks succeed, say:
"Your chat is linked and you confirmed the bot answered. Telegram setup is
complete." Stored status keeps asking for this human check on a later resume;
it cannot attest that she saw a reply or that a token has not since been revoked.
