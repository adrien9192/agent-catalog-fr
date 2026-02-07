# Script quotidien : Ajout de 3-5 nouveaux cas d'usage

## Comment utiliser

Lancez Claude Code dans le dossier du projet et utilisez cette commande :

```bash
claude -p "$(cat scripts/add-use-cases-prompt.md)"
```

Ou exécutez directement dans Claude Code :

```
/add-use-cases
```

## Fréquence recommandée

- 3-5 nouveaux cas d'usage par jour
- Variez les fonctions (Support, Sales, RH, Marketing, Finance, IT, Legal, Operations, Product)
- Alternez les difficultés (Facile, Moyen, Expert)
- Couvrez différents secteurs

## Automatisation avec cron

Pour automatiser l'exécution quotidienne :

```bash
# Ajouter au crontab (exécution à 6h chaque matin)
0 6 * * * cd /Users/digitalaine/Desktop/Website\ -\ Agent && npx claude -p "$(cat scripts/add-use-cases-prompt.md)" --allowedTools "Edit,Read,Write,Bash,Glob,Grep,WebSearch" 2>&1 >> scripts/logs/daily-$(date +\%Y\%m\%d).log
```

## Structure des fichiers

- `scripts/add-use-cases-prompt.md` - Le prompt pour Claude Code
- `scripts/add-use-cases.md` - Ce fichier (documentation)
- `src/data/use-cases.ts` - Le fichier contenant tous les cas d'usage
- `src/data/types.ts` - Les types TypeScript à respecter
