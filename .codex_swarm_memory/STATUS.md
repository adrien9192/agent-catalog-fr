# STATUS.md — État Courant

## État actuel
- **Phase** : Itération 3 — QMD Améliorations continues
- **Conformité 22 questions** : 16/22 DONE, 5 PARTIAL, 1 MISSING
- **Site live** : https://agent-catalog-fr.vercel.app
- **GitHub** : https://github.com/adrien9192/agent-catalog-fr

## Résumé Itérations

| Itération | Statut | Livraisons clés |
|-----------|--------|----------------|
| 1. MVP | ✅ DONE | 10 use cases, 41 pages, Playwright 36/36, Vercel |
| 2. Search+Newsletter+SEO | ✅ DONE | Fuse.js, Brevo, JSON-LD, OG images, sitemap |
| 3. QMD Améliorations | 🔄 EN COURS | Gaps Q6, Q13, Q14, Q16, Q17, Q22 |

## Gaps restants (Itération 3)

| Priorité | Item | Ref |
|----------|------|-----|
| 🔴 HIGH | Workflows n8n/Make concrets dans tutoriels | Q16 |
| 🔴 HIGH | Sections enterprise systématiques (10/10 UC) | Q17 |
| 🟡 MED | Séquence onboarding 5 emails Brevo | Q13 |
| 🟡 MED | Search highlights dans résultats | Q6 |
| 🟡 MED | Template visuel différencié par difficulté | Q14 |
| 🟡 MED | Playwright tests search/filters/signup | Q22 |
| 🟢 LOW | Images visuelles (hero, cards, illustrations) | Design |

## Newsletter Brevo — Vérifié
- Welcome email : ✅ delivered (12:18 UTC)
- Daily campaign #14 : ✅ sent, 2 delivered
- List ID : 3 ("Newsletter — L'usine à Agents IA")
- Sender : adrienlaine91@gmail.com (vérifié)
- Cron : 7h UTC daily via vercel.json

---
*Mis à jour : 2026-02-07*
