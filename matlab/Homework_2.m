A=[1 2; 0 1];
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
A