/** Cosmetic vehicle silhouettes; all use identical race physics. */
export const VEHICLES = [{id: 'formula', name: 'Formula'}, {id: 'hatchback', name: 'Hatchback'}, {id: 'motorbike', name: 'Motorbike'}, {id: 'camper', name: 'Camper van'}];
export const CAR_COLORS = ['#C66E7A', '#5C9FA0', '#99AD87', '#D0AF68', '#9E83B4', '#C58B64'];
export const COLOR_NAMES = ['Rose', 'Teal', 'Sage', 'Ochre', 'Plum', 'Terracotta'];
export function drawVehicle(ctx, vehicle, color, length) {
    const width = length * .44;
    if (vehicle === 'camper') {
        // A two-tone little camper, with a surfboard strapped to its cream roof.
        const bodyWidth = width * 1.18;
        ctx.fillStyle = '#17131C';
        for (const x of [-.32, .24]) for (const side of [-1, 1]) {
            ctx.fillRect(x * length, side * bodyWidth * .5 - bodyWidth * .12, length * .17, bodyWidth * .24);
        }
        ctx.fillStyle = color; ctx.strokeStyle = '#20182B'; ctx.lineWidth = .6;
        ctx.beginPath(); ctx.roundRect(-length * .48, -bodyWidth * .5, length * .96, bodyWidth, length * .1); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#F1E8D6';
        ctx.beginPath(); ctx.roundRect(-length * .36, -bodyWidth * .37, length * .54, bodyWidth * .74, length * .05); ctx.fill();
        ctx.fillStyle = '#302E3C';
        ctx.fillRect(length * .25, -bodyWidth * .36, length * .13, bodyWidth * .72);
        ctx.fillRect(-length * .43, -bodyWidth * .25, length * .05, bodyWidth * .5);
        ctx.strokeStyle = '#5B5244'; ctx.lineWidth = .7;
        ctx.beginPath(); ctx.moveTo(-length * .24, -bodyWidth * .37); ctx.lineTo(-length * .24, bodyWidth * .37);
        ctx.moveTo(length * .06, -bodyWidth * .37); ctx.lineTo(length * .06, bodyWidth * .37); ctx.stroke();
        ctx.fillStyle = '#D0AF68'; ctx.strokeStyle = '#5B5244'; ctx.lineWidth = .55;
        ctx.beginPath(); ctx.ellipse(-length * .08, 0, length * .29, bodyWidth * .13, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(-length * .31, 0); ctx.lineTo(length * .15, 0); ctx.stroke();
        ctx.fillStyle = '#FAF9FC';
        for (const y of [-.37, .24]) ctx.fillRect(length * .41, y * bodyWidth, length * .04, bodyWidth * .13);
        return;
    }
    if (vehicle === 'hatchback') {
        ctx.fillStyle = '#17131C';
        for (const x of [-.3, .25]) for (const y of [-.5, .35]) ctx.fillRect(x * length, y * width, length * .16, width * .18);
        ctx.fillStyle = color; ctx.strokeStyle = '#20182B'; ctx.lineWidth = .6;
        ctx.beginPath(); ctx.roundRect(-length * .46, -width * .45, length * .92, width * .9, length * .1); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#241A31';
        ctx.fillRect(length * .12, -width * .34, length * .15, width * .68);
        ctx.fillRect(-length * .29, -width * .32, length * .12, width * .64);
        ctx.fillStyle = '#FAF9FC';
        for (const y of [-.35, .23]) ctx.fillRect(length * .37, y * width, length * .05, width * .12);
        return;
    }
    if (vehicle === 'motorbike') {
        ctx.fillStyle = '#17131C';
        ctx.fillRect(-length * .49, -width * .13, length * .24, width * .26);
        ctx.fillRect(length * .26, -width * .13, length * .24, width * .26);
        ctx.strokeStyle = '#20182B'; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(length * .18, -width * .42); ctx.lineTo(length * .18, width * .42); ctx.stroke();
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.ellipse(0, 0, length * .35, width * .29, 0, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#241A31';
        ctx.beginPath(); ctx.ellipse(-length * .08, 0, length * .15, width * .24, 0, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#FAF9FC';
        ctx.beginPath(); ctx.arc(length * .09, 0, width * .16, 0, Math.PI * 2); ctx.fill();
        return;
    }
        // Four exposed tyres, separated from the narrow central body.
        ctx.fillStyle = '#17131C';
        for (const axle of [-.30, .25]) {
            for (const side of [-1, 1]) ctx.fillRect(axle * length - length * .11,
                side * width * .44 - width * .16, length * .22, width * .32);
        }
        ctx.fillStyle = color; ctx.strokeStyle = '#20182B'; ctx.lineWidth = .5;
        ctx.beginPath();
        ctx.moveTo(-length * .4, -width * .20);
        ctx.lineTo(-length * .15, -width * .30);
        ctx.lineTo(length * .13, -width * .16);
        ctx.lineTo(length * .46, -width * .07);
        ctx.lineTo(length * .46, width * .07);
        ctx.lineTo(length * .13, width * .16);
        ctx.lineTo(-length * .15, width * .30);
        ctx.lineTo(-length * .4, width * .20);
        ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.fillRect(length * .39, -width * .57, length * .08, width * 1.14);
        ctx.fillRect(-length * .48, -width * .46, length * .10, width * .92);
        // Cockpit and halo keep the silhouette recognisably single-seater.
        ctx.fillStyle = '#241A31';
        ctx.beginPath(); ctx.ellipse(-length * .08, 0, length * .12, width * .18, 0, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#F6F1FA'; ctx.lineWidth = .65;
        ctx.beginPath(); ctx.ellipse(-length * .06, 0, length * .09, width * .16, 0, -Math.PI / 2, Math.PI / 2); ctx.stroke();

}
