# AnimaLoraStudio Design System

This document is the durable visual contract for Studio Web. Product behavior and
information architecture remain authoritative in the application and product docs;
this file defines how those behaviors are presented consistently.

## 1. Direction

AnimaLoraStudio is a focused creative workbench, not a marketing surface. Its visual
world is warm ivory, restrained orange, precise typography, and compact technical
controls. Light and dark themes express the same hierarchy. Changes should evolve
this world rather than replace it.

The interface serves two audiences at once:

- New users need clear hierarchy, familiar controls, and visible next actions.
- Experienced users need dense parameter editing, fast scanning, and stable layouts.

Consistency means equal semantics receive equal treatment. It does not mean every
surface has identical density.

## 2. Sources of truth

| Layer | Authority | Responsibility |
| --- | --- | --- |
| Foundation | `studio/web/src/styles/tokens.css` | Color, type, spacing, radius, shadow, motion, control states |
| Utility bridge | `studio/web/tailwind.config.js` | Maps CSS tokens into Tailwind utilities |
| Primitives | `studio/web/src/components/Button.tsx`, `Badge.tsx` | Typed, accessible component APIs |
| Patterns | `PageHeader`, `StepShell`, `Dialog`, `Toast`, `Field` | Repeated page and interaction structures |
| Product surfaces | `studio/web/src/pages/` | Business state and composition, not new visual primitives |

A page may compose primitives with layout utilities. It must not recreate an
existing primitive with arbitrary colors, padding, font sizes, or hover states.

## 3. Foundations

### Color

Use semantic tokens instead of literal colors:

- `canvas` is the page field.
- `surface` is the normal content plane.
- `sunken` is for wells, navigation, and code/data regions.
- `elevated` is for overlays and popovers.
- `accent` identifies the primary action or active process.
- `ok`, `warn`, `err`, and `info` communicate state, never decoration alone.

Do not rely on color as the only state cue. Pair it with text, an icon, or an
indicator. Dark mode must preserve hierarchy rather than simply invert colors.

### Typography

- `text-base`: normal interface and body copy.
- `text-sm`: compact controls and secondary content.
- `text-xs`: metadata, timestamps, and badges.
- `text-2xs`: dense supporting labels only; never primary actions.
- `text-xl` and above: page or major section hierarchy.
- `font-mono`: code, paths, identifiers, logs, and measurements—not generic UI chrome.
- `.tnum`: changing or comparable numeric values.

Use the configured type scale. Arbitrary pixel font sizes require a documented
layout constraint and should remain exceptional.

### Spacing, radius, and depth

Use the `--s-*`, `--r-*`, and `--sh-*` scales. Within one hierarchy level, use one
radius: controls use `--r-md`, ordinary cards use `--r-lg`, and pills use
`--r-pill`. Dense workbench panels may use `--r-md` when their compactness is part
of the information structure.

Shadows communicate elevation. Borders communicate grouping. Do not add shadows
merely to decorate every container.

### Density

- **Default** is the baseline for navigation, settings, dialogs, and ordinary forms.
- **Compact** is allowed for parameter-heavy workbenches, tables, and repeated row
  controls. It must use an explicit small component size or the global tight density,
  not local pixel values.
- **Loose** increases readability without changing component semantics.

## 4. Button contract

Use `Button` for button elements. Links that navigate may use `buttonClassName()`
when they need button presentation while retaining link semantics.

| Variant | Use | Do not use for |
| --- | --- | --- |
| `primary` | The single leading action in a local decision scope | Every positive action on a page |
| `secondary` | Normal actions and alternate choices | Passive navigation or icon-only chrome |
| `ghost` | Low-emphasis actions, toolbar controls, dismissals | Destructive actions without another cue |
| `warning` | Interrupting or canceling reversible/in-progress work | Permanent deletion |
| `danger` | Irreversible deletion or discarding recoverable state | Routine cancellation |

Sizes:

- `md`: ordinary forms and dialogs.
- `sm`: headers, cards, and compact action rows.
- `xs`: dense tables, filters, and micro toolbars; never body copy squeezed smaller.

Rules:

- Button labels name the action.
- Icon-only buttons require an accessible label.
- Loading buttons remain labeled, expose `aria-busy`, and cannot be activated.
- Toggle buttons expose `aria-pressed`.
- Disabled, hover, active, and keyboard-focus states come from the primitive.
- Do not combine `bg-*`, `border-*`, and `text-*` to reinvent an existing variant.

## 5. Badge contract

Badges are non-interactive labels. A clickable pill is a button or link, not a badge.

| Tone | Meaning | Common states |
| --- | --- | --- |
| `neutral` | Passive, queued, unknown, or canceled metadata | pending, scheduled, canceled |
| `accent` | Active work or the current process | running, training, evaluating |
| `success` | Successful completion or confirmed availability | done, completed, available |
| `warning` | Paused, partial, or attention required | paused, partial |
| `danger` | Failure or invalid state | failed, error |
| `info` | Informational classification without success/failure | source or category labels |

An active badge may include the shared pulsing indicator. Use `sm` only for dense
metadata such as announcement tags; status badges use the default size.

Domain components such as `VersionStatusBadge` own the mapping from backend state to
badge tone. The generic `Badge` component must not know API enums.

## 6. Forms and help text

Default forms use the standard `.input` surface. Parameter-heavy schema forms may
use their existing compact canvas treatment as an explicit density variant.

Settings explanations belong in the label-adjacent `InfoButton` tooltip according
to `docs/design/ui-info-design.md`; do not add permanent explanatory paragraphs
under individual settings.

## 7. Accessibility and resilience

Every primitive must work with keyboard focus, disabled state, light/dark themes,
all three density modes, and both Chinese and English labels. Focus is always visible.
Controls must tolerate longer English copy without fixed-width truncation unless the
full value is available through an established disclosure pattern.

Respect `prefers-reduced-motion`; status information must remain understandable when
animation is disabled.

## 8. Migration policy

Migration is incremental:

1. Preserve the CSS classes as compatibility primitives.
2. Use typed components for new work.
3. Migrate representative surfaces with tests.
4. Move remaining pages by component family, not by arbitrary page batches.
5. Remove old compatibility paths only after repository-wide adoption.

A migration must preserve business behavior, route contracts, schema, SSE events,
and localization unless those changes are explicitly part of its scope.
