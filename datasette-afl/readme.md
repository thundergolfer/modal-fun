# 'AFL Data' Datasette app on Modal.com

> In progress...

The rough goal of this project is to make the answering of AFL stats questions much easier for myself
(and eventually others). Go on any [reddit.com/r/AFL/](https://www.reddit.com/r/AFL/) match thread during the AFL season and you'll find it littered with questions of the forms:

- When was the last time ...?
- Has X, which just happened, ever happened before?
- Who was the player who did X Ys during a game a few years ago?

Many of these stats questions can be answered with SQL tables containing up-to-date information on all
AFL-era (1990-present) games and player stats. So that's where I'll start.

**Rough example**

Which player scored the most goals in 2021?

```sql
SELECT player_name, team_name, SUM(goals) as total_goals
FROM match
WHERE year = 2021
GROUPY BY player_name
ORDER BY SUM(goals)
LIMIT 1
```
