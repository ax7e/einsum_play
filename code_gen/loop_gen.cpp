#include <bits/stdc++.h>

using namespace std;

//TODO: storage is also tunable
//TODO: tile and fusion may speed up the application
const string code1 = "void $0(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;"
    "lop($1)lop($2)lop($3)lop($4)lop($5)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}\n";
const string code2 = "CHECK_TIME( $0(TENSOR_DATA), \"$1\");\n";
string Replace(const string& str, const string& sub, const string& mod) {
    return regex_replace(str, regex(sub), mod);
}

int main(){
    FILE *doth=fopen("code.h","w");
    FILE *dotc=fopen("code.c","w");
    int id[]={0,1,2,3,4};
    string a[]={"i,0,I","j,0,J","k,0,K","l,0,L","m,0,M"};
    string b[]={"i","j","k","l","m"};
    do{
        string name;
        for(int i=0;i<5;++i)name+=b[id[i]];
        //cout<<name<<endl;
        string o1=Replace(code1,"\\$0",name);
        //cout<<name<<endl;
        for(int i=0;i<5;++i)o1=Replace(o1,"\\$"+to_string(i+1),a[id[i]]);
        //cout<<name<<endl;
        string o2=Replace(code2,"\\$0",name);
        o2=Replace(o2,"\\$1",name);
        cout<<o1<<endl;
        //cout<<o2<<endl;
        //break;
        fprintf(doth,"%s",o1.c_str());
        fprintf(dotc,"%s",o2.c_str());
    }while(next_permutation(id,id+5));

}