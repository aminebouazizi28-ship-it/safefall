// SafeFall – logique de l'application
// Réglages : seuil d'alerte (thr), formule du score (fonction frame), confirmation au sol (still/3)

const $=s=>document.querySelector(s);
const sig=x=>1/(1+Math.exp(-x)),lerp=(a,b,k)=>a+(b-a)*k;
const sm=(x,a,b)=>{x=Math.min(1,Math.max(0,(x-a)/(b-a)));return x*x*(3-2*x)};

// Les chambres (modifiez les noms ici)
const rooms=[
{n:"Chambre 1",r:"Habiba, 84 ans",s:"Surveillance",p:0},
{n:"Chambre 2",r:"Mongi, 79 ans",s:"Au lit",p:1},
{n:"Chambre 3",r:"Zohra, 88 ans",s:"Dans le fauteuil",p:2},
{n:"Chambre 4",r:"Salah, 81 ans",s:"Au repos",p:1}];

let sel=0,scn="walk",t0=0,simT=0,last=0,camOn=true,thr=.8;
let hist=[],series=[],lastS=0,shown=0,prevH=1.55,vs=0,still=0,logs=[];

// Ajoute une ligne dans le journal
function say(m){const d=new Date().toLocaleTimeString("fr-FR");logs.unshift("<time>"+d+"</time>"+m);logs=logs.slice(0,7);$("#log").innerHTML=logs.map(x=>"<li>"+x+"</li>").join("")}

// Change de scénario (marche, assise, se baisser, chute)
function setScn(s){scn=s;t0=simT;hist=[];vs=0;still=0;document.querySelectorAll("#scn button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.s===s))}

// Affiche la liste des chambres et la bannière d'alerte
function renderRooms(){
 $("#rooms").innerHTML=rooms.map((r,i)=>{
  const al=r.alert&&!r.ack;
  const p=i===sel?Math.round(shown*100):r.p;
  return '<button class="room'+(al?" al":"")+'" data-i="'+i+'" aria-current="'+(i===sel)+'"><strong>'+r.n+'</strong><span>'+r.r+'</span><span class="st">'+(al?"Chute probable · "+Math.round(r.alert.p*100)+" %":r.s+" · "+p+" %")+'</span></button>'}).join("");
 const a=rooms.find(r=>r.alert&&!r.ack),b=$("#banner");
 if(a){b.classList.add("on");$("#bt").textContent="Chute probable – "+a.n+" · "+a.r.split(",")[0]+" · "+Math.round(a.alert.p*100)+" %";$("#bs").textContent="SMS envoyé à l’infirmière de garde et message WhatsApp envoyé à la famille."}
 else b.classList.remove("on");
 $("#stTitle").textContent=rooms[sel].n+" – "+rooms[sel].r;
}

// Clics sur les boutons
$("#rooms").onclick=e=>{const b=e.target.closest(".room");if(!b)return;sel=+b.dataset.i;const r=rooms[sel];setScn(r.alert&&!r.ack?"fall":"walk");if(r.alert&&!r.ack)t0=simT-5;renderRooms()};
$("#scn").onclick=e=>{const b=e.target.closest("button");if(b){rooms[sel].alert=null;rooms[sel].ack=false;setScn(b.dataset.s);renderRooms()}};
$("#mode").onclick=e=>{const b=e.target.closest("button");if(!b)return;
 if(b.id==="camTog"){camOn=!camOn;b.setAttribute("aria-pressed",!camOn);b.textContent=camOn?"Désactiver la caméra (soins)":"Réactiver la caméra"}
};
$("#thr").oninput=e=>{thr=e.target.value/100;$("#thv").textContent=e.target.value+" %";$("#pt").style.left=e.target.value+"%"};
$("#ack").onclick=()=>{const r=rooms.find(r=>r.alert&&!r.ack);if(!r)return;r.ack=true;r.alert=null;say(r.n+" : personne prise en charge.");if(rooms[sel]===r)setScn("walk");renderRooms()};

// ---- 1. Le corps simulé : position du patient selon le scénario
function pose(s,t){
 let hx=1.6,hh=.9,a=3,ld=.05,sw=0;
 if(s==="walk"){hx=1.6+.15*Math.sin(.5*t);a=4+2*Math.sin(3.2*t);hh=.9+.02*Math.sin(6.4*t);sw=.2*Math.sin(3.2*t)}
 if(s==="sit"){let k=sm(t,.2,3)*(1-sm(t,7,9.5));hh=lerp(.9,.45,k);a=lerp(3,8,k);ld=lerp(.05,.4,k)}
 if(s==="bend"){let k=sm(t,.2,.7)*(1-sm(t,1.6,2.8));hh=lerp(.9,.55,k);a=lerp(3,78,k);ld=lerp(.05,.3,k)}
 if(s==="fall"){let k=sm(t,1,1.6);k=k*k;hh=lerp(.9,.12,k);a=lerp(3,90,k);ld=lerp(.05,-.85,k);hx=1.6+.4*k;sw=t<1?.1*Math.sin(8*t):0}
 return{hx,hh,a,ld,sw}
}

// ---- 2. Squelette : calcul des genoux (cinématique inverse à 2 segments)
function leg(h,f){
 const dx=f[0]-h[0],dy=f[1]-h[1],d0=Math.hypot(dx,dy)||1,d=Math.min(.89,d0),hh=Math.sqrt(Math.max(0,.2025-d*d/4));
 let p=[-dy/d0,dx/d0];if(p[0]+.5*p[1]<0)p=[-p[0],-p[1]];
 return[(h[0]+f[0])/2+p[0]*hh,(h[1]+f[1])/2+p[1]*hh]
}
function skel(p){
 const r=p.a*Math.PI/180,u=[Math.sin(r),Math.cos(r)],hip=[p.hx,p.hh],sh=[hip[0]+.55*u[0],hip[1]+.55*u[1]],head=[sh[0]+.27*u[0],sh[1]+.27*u[1]];
 const fA=[p.hx+p.ld+p.sw,0],fB=[p.hx+p.ld*.6-p.sw,0];
 const el=[sh[0]-.28*Math.sin(r+.08+p.sw),sh[1]-.28*Math.cos(r+.08+p.sw)],hd=[el[0]-.27*Math.sin(r+.3+p.sw*1.5),el[1]-.27*Math.cos(r+.3+p.sw*1.5)];
 return{head,sh,hip,el,hd,kA:leg(hip,fA),fA,kB:leg(hip,fB),fB}
}

const cam=$("#cam").getContext("2d"),sp=$("#spark").getContext("2d");
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const PX=(x,y)=>[50+x*150,318-y*150];

// ---- 3. Affichage de la caméra : squelette, cadre et pourcentage
function draw(S,alert,prob){
 const c=cam,W=640,H=360;c.clearRect(0,0,W,H);
 const ink=css("--ink"),mut=css("--mut"),col=alert?css("--al"):css("--ok");
 c.strokeStyle=mut;c.fillStyle=mut;c.globalAlpha=.35;c.lineWidth=2;
 c.beginPath();c.moveTo(0,318);c.lineTo(W,318);c.stroke();
 c.strokeRect(420,70,110,120);c.strokeRect(14,276,150,28);c.strokeRect(14,244,12,60);c.globalAlpha=1;
 if(!camOn){c.fillStyle=mut;c.font="600 18px Bricolage Grotesque,sans-serif";c.textAlign="center";c.fillText("Caméra désactivée pendant les soins",W/2,H/2);c.textAlign="left";return}
 const xs=[],ys=[];
 const lines=[["head","sh"],["sh","hip"],["sh","el"],["el","hd"],["hip","kA"],["kA","fA"],["hip","kB"],["kB","fB"]];
 c.strokeStyle=col;c.lineWidth=4;c.lineCap="round";
 lines.forEach(([a,b])=>{const A=PX(...S[a]),B=PX(...S[b]);c.beginPath();c.moveTo(...A);c.lineTo(...B);c.stroke()});
 c.fillStyle=ink;
 Object.values(S).forEach(j=>{const q=PX(...j);xs.push(q[0]);ys.push(q[1]);c.beginPath();c.arc(q[0],q[1],4,0,7);c.fill()});
 const x0=Math.min(...xs)-16,x1=Math.max(...xs)+16,y0=Math.min(...ys)-24,y1=Math.max(...ys)+8;
 c.strokeStyle=col;c.lineWidth=2;c.setLineDash([6,4]);c.strokeRect(x0,y0,x1-x0,y1-y0);c.setLineDash([]);
 c.fillStyle=col;c.fillRect(x0,y0-22,126,22);c.fillStyle="#fff";c.font="600 13px Bricolage Grotesque,sans-serif";
 c.fillText((alert?"Chute ":"Personne ")+Math.round(prob*100)+" %",x0+8,y0-6);
 if(!matchMedia("(prefers-reduced-motion:reduce)").matches){const sx=(simT*90)%W;c.strokeStyle=css("--ok");c.globalAlpha=.25;c.lineWidth=2;c.beginPath();c.moveTo(sx,0);c.lineTo(sx,H);c.stroke();c.globalAlpha=1}
}

// Petite courbe de la probabilité sur les 30 dernières secondes
function drawSpark(){
 const c=sp,W=300,H=60;c.clearRect(0,0,W,H);
 c.strokeStyle=css("--ln");c.lineWidth=1;const ty=H-thr*H;c.beginPath();c.moveTo(0,ty);c.lineTo(W,ty);c.stroke();
 c.strokeStyle=css("--ok");c.lineWidth=2;c.beginPath();
 series.forEach((v,i)=>{const x=i*W/150,y=H-2-v*(H-4);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()
}

// ---- 4. Boucle principale (60 fois par seconde) : mesures, probabilité, alerte
function frame(ts){
 const dt=Math.min(.05,(ts-last)/1000||.016);last=ts;simT+=dt;const t=simT-t0;
 if((scn==="sit"&&t>9.6)||(scn==="bend"&&t>2.9))setScn("walk");
 const p=pose(scn,simT-t0),S=skel(p),headH=S.head[1];

 // Vitesse de descente de la tête (lissée)
 vs=vs*.6+(-(headH-prevH)/dt)*.4;prevH=headH;
 hist.push({t:simT,v:Math.max(0,vs)});
 while(hist[0]&&simT-hist[0].t>2.5)hist.shift();
 const vmax=Math.max(0,...hist.filter(h=>simT-h.t<1.5).map(h=>h.v));

 // Score de mouvement : vitesse de descente, angle du tronc, hauteur de la tête
 const sc=camOn?sig(1.8*(vmax-1)+.07*(p.a-50)+2*(.7-headH)):0,conf=Math.min(1,still/3);

 // Confirmation : sans immobilité au sol, la probabilité est divisée par deux
 let f=sc*(.5+.5*conf);

 // Déclenchement de l'alerte
 const r=rooms[sel];
 if(f>=thr&&!r.alert&&!r.ack){r.alert={p:f,esc:false};say(r.n+" : chute probable ("+Math.round(f*100)+" %). SMS et WhatsApp envoyés.");renderRooms()}
 if(r.alert&&f>r.alert.p)r.alert.p=f;
 shown+=(((r.alert&&!r.ack)?Math.max(f,r.alert.p):f)-shown)*.25;

 // Immobilité au sol (compteur)
 const moving=Math.abs(vs)>.15;
 if(p.a>60&&headH<.6&&!moving)still+=dt;else if(moving)still=0;
 if(r.alert&&!r.ack&&still>10&&!r.alert.esc){r.alert.esc=true;say(r.n+" : toujours au sol après 10 s. Appel du médecin.")}

 // Affichage
 draw(S,shown>=thr,shown);
 $("#pv").textContent=Math.round(shown*100);
 const pb=$("#pb");pb.style.width=shown*100+"%";pb.style.background=shown>=thr?css("--al"):shown>thr*.6?css("--wn"):css("--ok");
 $("#m1").textContent=p.a.toFixed(0)+"°";$("#m2").textContent=Math.max(0,vs).toFixed(1)+" m/s";$("#m3").textContent=headH.toFixed(2)+" m";
 $("#m4").textContent=Math.round(conf*100)+" %";$("#m5").textContent=camOn?Math.round(sc*100)+" %":"coupée";$("#m7").textContent=still.toFixed(0)+" s";
 if(simT-lastS>.2){lastS=simT;series.push(shown);if(series.length>150)series.shift();drawSpark();renderRooms()}
 requestAnimationFrame(frame)
}
renderRooms();requestAnimationFrame(frame);
