import type { Battle } from "../../api/playerAnalysis";
import { percent } from "../../lib/format";
import { facedMost } from "../../lib/stats";
import { CardImage } from "./CardImage";

function FacedMost({ battles, total }: { battles: Battle[]; total: number }) {
  const rows = facedMost(battles);
  if (rows.length === 0) return null;
  return (
    <table className="faced-most">
      <caption>Faced most</caption>
      <thead>
        <tr>
          <th scope="col">
            <span className="visually-hidden">Card</span>
          </th>
          <th scope="col">Faced</th>
          <th scope="col">Win chance</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ card, faced, winChance }) => (
          <tr key={card.name}>
            <th scope="row">
              <span className="faced-icon">
                <CardImage card={card} decorative />
              </span>
              {card.name}
            </th>
            <td className="num">
              {faced}/{total}
            </td>
            <td className="num">{percent(winChance)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export { FacedMost };
