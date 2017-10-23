/****************
 * User Controlled Topologically Adaptive Snakes.
 * T-Snakes are an idea proposed by McInerney and Terzopoulos in the
 * mid 90s. This is a simple buggy prototype implementation.
 *
 *
 * (c) 2002, Andreas Bærentzen
 *************************************/

#include <vector>
#include <GEL/Util/Grid2D.h>
#include <GEL/CGLA/Vec2f.h>
#include <GEL/CGLA/Vec2i.h>
#include <GEL/CGLA/Mat2x2f.h>

//#include <GEL/GL/glew.h>
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h> // Header File For The OpenGL32 Library
#include <OpenGL/glu.h> // Header File For The GLu32 Library
#include <GLUT/glut.h>

#include "image.h"

#include "profile.h"

using namespace CGLA;
using namespace Util;
using namespace std;

namespace
{
    int GRIDX = 30;
    int GRIDY = 30;
    int GRID_SCALE = 3;
    
    int win_size_x = 500;
    int win_size_y = 500;
    
    bool SHOW_NORMALS = false;
    
    image m_image;
    
    // ----------------------------------------------------------------------
    // Vertex data base. Simple array wrapper for 2D vector arrays.
    // Has single instance. Should be a singleton.
    
    class VertexDB
    {
        vector<Vec2f> vertices;
        vector<Vec2f> normals;
        
        inline Vec2f rand_vec()
        {
            const float len = 1e-4;
            return len*Vec2f(2,1);
        }
        
    public:
        
        int add(const Vec2f& p)
        {
            int idx = vertices.size();
            vertices.push_back(p);
            normals.push_back(Vec2f(0));
            return idx;
        }
        
        const Vec2f& get(int i) const
        {
            assert(i>=0 && i < vertices.size());
            return vertices[i];
        }
        
        void set_normal(int i, const Vec2f& n)
        {
            assert(i>=0 && i < vertices.size());
            normals[i] = n;
        }
        
        void add2_normal(int i, const Vec2f& n)
        {
            assert(i>=0 && i < vertices.size());
            normals[i] += n;
            normals[i].normalize();
        }
        
        void clear()
        {
            vertices.clear();
        }
        
        void evolute_curve()
        {
            static int iter = 0;
            double dt = 0.7;
//            if(iter++ > 100)
//                dt = 0.1;
            for (int i = 0; i < vertices.size(); i++)
            {
                auto pos = vertices[i];
                auto n = normals[i];
                auto inten = m_image.get_intensity_scale(pos[0],pos[1]);
                
                auto f = n*(inten - 0.25)*(-dt);
                
                vertices[i] = pos + f;
            }
        }
        
        void transform(const Vec2f& trans, const Mat2x2f& mat)
        {
            for(int i=0;i<vertices.size();++i)
            {
                vertices[i] -= trans;
                vertices[i] = mat*vertices[i];
                vertices[i] += trans;
            }
        }
        
        void shake()
        {
            for(int i=0;i<vertices.size();++i)
                if(vertices[i]==Vec2f(vertices[i]))
                    vertices[i] += rand_vec();
        }
        
        void dilate(float x)
        {
            for(int i=0;i<vertices.size();++i)
                vertices[i] += normals[i] * x;
        }
        
        void dilate(int x, int y, float s0)
        {
            Vec2f p(x/float(GRID_SCALE), GRIDY-y/float(GRID_SCALE));
            for(int i=0;i<vertices.size();++i)
            {
                float s = s0*exp(-sqr_length(vertices[i]-p));
                vertices[i] += normals[i] * s;
            }
        }
    };
    
    
    // ----------------------------------------------------------------------
    // Vertex data base
    
    VertexDB VERTEX_DB;
    
    
    // ----------------------------------------------------------------------
    // Segment class
    
    class Segment
    {
    public:
        int p0i,p1i;
        
        const Vec2f& p0() const
        {
            assert(p0i>=0);
            return VERTEX_DB.get(p0i);
        }
        
        const Vec2f& p1() const
        {
            assert(p1i>=0);
            return VERTEX_DB.get(p1i);
        }
        
        Segment(): p0i(-1), p1i(-1) {}
        
        
        Segment(int _p0i, int _p1i):
        p0i(_p0i), p1i(_p1i) {}
        
        void swap_vertices()    { swap(p0i,p1i);    }
        
        void gl_draw()
        {
            glBegin(GL_LINES);
            glVertex2fv(p0().get());
            glVertex2fv(p1().get());
            if(SHOW_NORMALS)
            {
                Vec2f m = p0() + (p1()-p0())/2.0f;
                glVertex2fv(m.get());
                m += get_normal();
                glVertex2fv(m.get());
            }
            
            glEnd();
        }
        
        const Vec2f get_normal() const
        {
            return normalize(orthogonal(p0()-p1()));
        }
        
        
        
        void get_bbox(float&xmin, float&xmax,
                      float&ymin, float&ymax,
                      float&dmin, float&dmax)
        {
            const Vec2f DIAG(1,-1);
            xmin = min(p0()[0],p1()[0]);
            xmax = max(p0()[0],p1()[0]);
            ymin = min(p0()[1],p1()[1]);
            ymax = max(p0()[1],p1()[1]);
            dmin = min(dot(DIAG, p0()),dot(DIAG, p1()));
            dmax = max(dot(DIAG, p0()),dot(DIAG, p1()));
        }
        
        void intersect(const Vec2f& p, const Vec2f& d, Vec2f& pi)
        {
            Vec2f n = normalize(orthogonal(p0()-p1()));
            float dist = dot(n, p0()-p);
            pi = p + d * dist/dot(d,n);
        }
        
    };
    
    
    // ----------------------------------------------------------------------
    // Segment database
    // Again. this should be a singleton.
    
    vector<Segment> SEGMENT_DB;
    
    
    // ----------------------------------------------------------------------
    // Edge class. This class represents a grid edge
    //
    
    struct Edge
    {
        int used;
        Vec2f pos;
        char sgn;
        int id;
        
        Edge(): used(false), id(-1) {}
        
        void set_pos(const Vec2f& p, char _sgn)
        {
            if(used>2) cout << __FILE__ << __LINE__ << endl;
            if(used>0)
            {
                if(sgn == _sgn)
                {
                    ++used;
                    return;
                }
                --used;
                return;
            }
            used = 1;
            pos = p;
            sgn = _sgn;
            id = -1;
        }
        
        int get_vertex_id()
        {
            if(!used) return -1;
            if(id==-1) id = VERTEX_DB.add(pos);
            return id;
        }
        
    };
    
    
    // ----------------------------------------------------------------------
    // Pixel class: Contains information about edges and whether it is
    // inside.
    
    struct Pixel
    {
        bool inside;
        Edge hedge, vedge, dedge;
    };
    
    
    // ----------------------------------------------------------------------
    // 2D Grid of pixels
    // Grid class
    
    typedef Grid2D<Pixel> GridType;
    
    // ----------------------------------------------------------------------
    // Edge intersector function.
    // This function finds all the intersections of a segment and the grid edges.
    
    void edge_intersector(GridType& grid, Segment& seg)
    {
        float xmin, xmax, ymin, ymax, dmin, dmax;
        seg.get_bbox(xmin,xmax, ymin, ymax, dmin, dmax);
        Vec2f seg_norm = seg.get_normal();
        
        
        for(int i=int(ceilf(xmin));i<xmax; ++i)
        {
            Vec2f p(i,ymin);
            Vec2f d(0,1);
            Vec2f pi;
            
            float dotprod = dot(seg_norm,d);
            char sgn = (dotprod>0)?1:-1;
            
            seg.intersect(p,d,pi);
            grid(Vec2i(pi)).vedge.set_pos(pi,sgn);
        }
        
        for(int i=int(ceilf(ymin));i<ymax; ++i)
        {
            Vec2f p(xmin,i);
            Vec2f d(1,0);
            Vec2f pi;
            
            float dotprod = dot(seg_norm,d);
            char sgn = (dotprod>0)?1:-1;
            
            seg.intersect(p,d,pi);
            grid(Vec2i(pi)).hedge.set_pos(pi,sgn);
        }
        
        for(int i=int(ceilf(dmin));i<dmax; ++i)
        {
            Vec2f p(i,0);
            Vec2f d = normalize(Vec2f(1,1));
            Vec2f pi;
            
            float dotprod = dot(seg_norm,d);
            char sgn = (dotprod>0)?1:-1;
            
            seg.intersect(p,d,pi);
            grid(Vec2i(pi)).dedge.set_pos(pi,sgn);
        }
        
    }
    
    // ----------------------------------------------------------------------
    // Process snake. Loops over all segments and computes the intersection
    // of the snake and the edges of the grid.
    
    void process_snake(GridType& grid)
    {
        VERTEX_DB.shake();
        for(int i=0;i<SEGMENT_DB.size();++i)
            edge_intersector(grid, SEGMENT_DB[i]);
    }
    
    void give_snake_normals()
    {
        for(int i=0;i<SEGMENT_DB.size();++i)
        {
            Vec2f n = SEGMENT_DB[i].get_normal();
            if(SEGMENT_DB[i].p0i >=0)
                VERTEX_DB.set_normal(SEGMENT_DB[i].p0i,n);
        }
        for(int i=0;i<SEGMENT_DB.size();++i)
        {
            Vec2f n = SEGMENT_DB[i].get_normal();
            if(SEGMENT_DB[i].p1i >=0)
                VERTEX_DB.add2_normal(SEGMENT_DB[i].p1i,n);
        }
    }
    
    void dilate_snake(float x)
    {
        VERTEX_DB.dilate(x);
    }
    
    // ----------------------------------------------------------------------
    // Process grid. Find inside outside information. Create new snake.
    
    const char VERTEX_CASES[] = {0,1,2,3,3,2,1,0};
    
    void process_grid(GridType& grid)
    {
        // First loop set inside outside information
        
        for (int j=0;j< GRIDY; ++j)
        {
            bool inside = false;
            for(int i=0; i<GRIDX; ++i)
            {
                grid(i,j).inside = inside;
                const Edge& hedge = grid(i,j).hedge;
                if(hedge.used)
                    inside = ! inside;
            }
        }
        
        VERTEX_DB.clear();
        vector<Segment> new_segs(0);
        for (int j=0;j< GRIDY-1; ++j)
        {
            for(int i=0; i<GRIDX-1; ++i)
            {
                int cse = int(grid(i,j).inside) +
                2 * int(grid(i,j+1).inside) +
                4 * int(grid(i+1,j+1).inside);
                
                int id0, id1;
                switch(VERTEX_CASES[cse])
                {
                    case 1:
                        id0 = grid(i,j).dedge.get_vertex_id();
                        id1 = grid(i,j).vedge.get_vertex_id();
                        if(id0>=0&&id1>=0)
                        {
                            new_segs.push_back(Segment(id0, id1));
                            if(cse/4==1) new_segs.back().swap_vertices();
                        }
                        break;
                    case 2:
                        id0 = grid(i,j).vedge.get_vertex_id();
                        id1 = grid(i,j+1).hedge.get_vertex_id();
                        if(id0>=0&&id1>=0)
                        {
                            new_segs.push_back(Segment(id0, id1));
                            if(cse/4==1) new_segs.back().swap_vertices();
                        }
                        break;
                    case 3:
                        id0 = grid(i,j).dedge.get_vertex_id();
                        id1 = grid(i,j+1).hedge.get_vertex_id();
                        if(id0>=0&&id1>=0)
                        {
                            new_segs.push_back(Segment(id0, id1));
                            if(cse/4==1) new_segs.back().swap_vertices();
                        }
                        break;
                }
                
                
                cse = int(grid(i,j).inside) +
                2 * int(grid(i+1,j).inside) +
                4 * int(grid(i+1,j+1).inside);
                
                switch(VERTEX_CASES[cse])
                {
                    case 1:
                        id0 = grid(i,j).dedge.get_vertex_id();
                        id1 = grid(i,j).hedge.get_vertex_id();
                        if(id0>=0&&id1>=0)
                        {
                            new_segs.push_back(Segment(id0, id1));
                            if(cse/4==0) new_segs.back().swap_vertices();
                        }
                        break;
                    case 2:
                        id0 = grid(i,j).hedge.get_vertex_id();
                        id1 = grid(i+1,j).vedge.get_vertex_id();
                        if(id0>=0&&id1>=0)
                        {
                            new_segs.push_back(Segment(id0, id1));
                            if(cse/4==0) new_segs.back().swap_vertices();
                        }
                        break;
                    case 3:
                        id0 = grid(i,j).dedge.get_vertex_id();
                        id1 = grid(i+1,j).vedge.get_vertex_id();
                        if(id0>=0&&id1>=0)
                        {
                            new_segs.push_back(Segment(id0, id1));
                            if(cse/4==0) new_segs.back().swap_vertices();
                        }
                        break;
                }
                
            }
        }
        SEGMENT_DB = new_segs;
    }
    
    
    
    void display()
    {
        profile t("sanke 1");
        GridType grid(GRIDX, GRIDY);
        process_snake(grid);
        process_grid(grid);
        t.done();
        
        profile t1("draw");
        glClearColor(1,1,1,1);
        glClear(GL_COLOR_BUFFER_BIT);
        
        
//        m_image.draw_image_scale();

        glLineWidth(5);
        glColor3f(1,0,0);
        for(int i=0;i<SEGMENT_DB.size(); ++i)
            SEGMENT_DB[i].gl_draw();

//        glLineWidth(1);
//
//        glPointSize(4);
//        for(int i=0;i<GRIDX; ++i)
//            for(int j=0; j< GRIDY; ++j)
//            {
//
//                glColor3f(0,1,0);
//                glBegin(GL_LINES);
//                if(grid(i,j).hedge.used) glColor3f(0,0,0);
//                glVertex2f(i,j);
//                glVertex2f(i+1,j);
//                glColor3f(0,1,0);
//
//                if(grid(i,j).dedge.used) glColor3f(0,0,0);
//                glVertex2f(i,j);
//                glVertex2f(i+1,j+1);
//                glColor3f(0,1,0);
//
//                if(grid(i,j).vedge.used) glColor3f(0,0,0);
//                glVertex2f(i,j);
//                glVertex2f(i,j+1);
//                glColor3f(0,1,0);
//                glEnd();
//
//                glBegin(GL_POINTS);
//                if(grid(i,j).inside)
//                {
//                    glColor3f(0,0,0);
//                    glVertex2f(i,j);
//                }
//                glColor3f(0,0,1);
//                if(grid(i,j).vedge.used)
//                    glVertex2fv(grid(i,j).vedge.pos.get());
//                if(grid(i,j).hedge.used)
//                    glVertex2fv(grid(i,j).hedge.pos.get());
//                if(grid(i,j).dedge.used)
//                    glVertex2fv(grid(i,j).dedge.pos.get());
//                glEnd();
//            }


        
        glFinish();
        glutSwapBuffers();
        
        
    }
    
    bool left_mouse=false;
    bool right_mouse=false;
    
    void key(unsigned char c,int x, int y)
    {
        profile t("snake");
        give_snake_normals();
        VERTEX_DB.evolute_curve();
        t.done();
        profile::close();
//        VERTEX_DB.transform(Vec2f(15,15),
//                            Mat2x2f(cos(.1),sin(.1),-sin(.1),cos(.1)));
        glutPostRedisplay();
    }
    
    void mouse(int x, int y)
    {
        give_snake_normals();
        
        if(left_mouse)
            VERTEX_DB.dilate(x,y,.2);
        if(right_mouse)
            VERTEX_DB.dilate(x,y,-.2);
        
        glutPostRedisplay();
    }
    
    void mouse2(int b, int s, int x, int y)
    {
        if(s==GLUT_UP)
        {
            left_mouse=false;
            right_mouse=false;
        }
        else
        {
            if(b==GLUT_LEFT_BUTTON)
                left_mouse=true;
            if(b==GLUT_RIGHT_BUTTON)
                right_mouse=true;
        }
    }
    
}



int main(int argc, char **argv)
{
    // Load image
    m_image.load_image("./Data/snake.png");
    
    win_size_x = m_image.width();
    win_size_y = m_image.height();
    
    GRIDX = (int)(m_image.width()/GRID_SCALE);
    GRIDY = (int)(m_image.height()/GRID_SCALE);
    
    m_image.image_scale[0] = (double)GRIDX/m_image.width();
    m_image.image_scale[1] = (double)GRIDY/m_image.height();
    
    // Init OpenGL
    
    glutInit(&argc, argv);
    glutInitWindowSize(win_size_x, win_size_y);
    glutInitDisplayMode(GLUT_ALPHA|GLUT_DOUBLE);
    glutCreateWindow("T-Snakes");
    
    m_image.set_gl_texture();
    
    glClearColor(1,1,1,1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,GRIDX,0,GRIDY);
    glViewport(0,0,win_size_x,win_size_y);
    
    glMatrixMode(GL_MODELVIEW);
    
    glutKeyboardFunc(key);
    glutDisplayFunc(display);
    glutMotionFunc(mouse);
    glutMouseFunc(mouse2);
    
    int p0,p1;
    int gap = 2.5;
    SEGMENT_DB.push_back(Segment(p0=VERTEX_DB.add(Vec2f(gap,gap)),
                                 p1=VERTEX_DB.add(Vec2f(GRIDX-gap,gap))));
    int p2;
    SEGMENT_DB.push_back(Segment(p1,
                                 p2=VERTEX_DB.add(Vec2f(GRIDX-gap,GRIDY-gap))));
    int p3;
    SEGMENT_DB.push_back(Segment(p2,
                                 p3=VERTEX_DB.add(Vec2f(gap,GRIDY - gap))));
    
    SEGMENT_DB.push_back(Segment(p3,p0));
    
    profile t("100");
    for (int i = 0; i < 200; i++)
    {
        give_snake_normals();
        VERTEX_DB.evolute_curve();
        
        GridType grid(GRIDX, GRIDY);
        process_snake(grid);
        process_grid(grid);
    }
    
    t.done();
    profile::close();
    
    glutMainLoop();
    return 0;
}

