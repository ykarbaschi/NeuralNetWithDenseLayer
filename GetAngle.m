function angle=GetAngle(a,b)
costheta = dot(a,b)/(norm(a)*norm(b));
angle = acos(costheta);