A=randi(1000,10,10);
rowcol=size(A);
veckholder=zeros(rowcol(2),1);
for i=1:rowcol(2)
    %display(A(:,i));
    if i>=2
        for j=1:i-1
            veckholder=veckholder-dot(A(:,j),A(:,i))/dot(A(:,j),A(:,j))*A(:,i-1);
        end
    else
        A(:,i)=A(:,i);
    end
    A(:,i)=A(:,i)+veckholder;
    
end

for k=1:rowcol(2)
    if norm(A(:,k))==0
        display('Division by zero')
    else
        A(:,k)=A(:,k)/norm(A(:,k));
    end
end
%A
A'*A
eig(A'*A)
det(A)